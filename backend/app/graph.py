import uuid
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy.ext.asyncio import AsyncSession

from app.states.collect import run_collect
from app.states.fetch_data import run_fetch_data
from app.states.generate import run_generate
from app.states.validate import run_validate
from app.states.retrieve import run_retrieve
from app.states.list_trips import format_trip_list
from app.storage import (
    load_trip,
    save_trip,
    get_itinerary,
    list_trips_for_user,
    find_trip_by_destination,
)

MAX_RETRIES = 3

_checkpointer = MemorySaver()


class GraphState(TypedDict, total=False):
    session_id: str
    user_id: str
    trip_state: dict
    trip_status: str
    last_user_message: str
    intent: str
    destination_hint: Optional[str]
    missing_fields: list
    assistant_reply: str
    ready: bool
    hotels: list
    weather: dict
    itinerary: dict
    validation_issues: list
    is_valid: bool
    retry_count: int
    new_session_id: Optional[str]


def build_graph(db: AsyncSession):

    async def collect_node(state: GraphState) -> dict:
        trip_state, status = await load_trip(db, state["session_id"], state["user_id"])
        result = await run_collect(trip_state, state["last_user_message"], status)
        updated_trip_state = result.updated_trip_state.model_dump()

        current_destination = trip_state.get("destination")
        new_destination = updated_trip_state.get("destination")

        if (
            result.intent == "new_or_modify"
            and status == "generated"
            and current_destination
            and new_destination
            and current_destination.lower() != new_destination.lower()
        ):
            new_session_id = str(uuid.uuid4())
            fresh_state = {
                "destination": new_destination,
                "start_date": updated_trip_state.get("start_date"),
                "end_date": updated_trip_state.get("end_date"),
                "members": updated_trip_state.get("members"),
                "budget": updated_trip_state.get("budget"),
                "interests": updated_trip_state.get("interests", []),
            }
            await save_trip(db, new_session_id, state["user_id"], fresh_state, status="collecting")
            reply = (
                f"Looks like you want to plan a new trip to {new_destination}! "
                f"I've started a fresh trip thread for that so your current trip stays untouched. "
                f"Continue there with session_id: {new_session_id}"
            )
            return {
                "trip_state": trip_state,
                "assistant_reply": reply,
                "new_session_id": new_session_id,
                "intent": "new_trip_started",
                "ready": False,
            }

        if result.intent == "new_or_modify":
            await save_trip(db, state["session_id"], state["user_id"], updated_trip_state, status="collecting")
            reply = (
                result.clarifying_question
                if result.missing_fields
                else "Got all the details I need."
            )
        elif result.intent == "destination_info":
            # Persist destination + shown-places context (status unchanged) so a
            # follow-up like "show me more" still knows what's already been said.
            await save_trip(db, state["session_id"], state["user_id"], updated_trip_state, status=status)
            reply = result.direct_reply or ""
        elif result.intent == "greeting":
            reply = result.direct_reply or "Hi! I can help you plan a trip -- where would you like to go?"
        elif result.intent == "farewell":
            reply = result.direct_reply or "Have a great trip! Come back anytime you want to plan another one."
        elif result.intent == "off_topic":
            reply = (
                result.direct_reply
                or "I'm a travel planning assistant -- I can help you plan a trip, check an existing itinerary, or list your past trips. What would you like to do?"
            )
        else:
            reply = ""

        return {
            "trip_state": updated_trip_state,
            "trip_status": status,
            "intent": result.intent,
            "destination_hint": result.destination_hint,
            "missing_fields": result.missing_fields,
            "assistant_reply": reply,
            "ready": not result.missing_fields,
        }

    # This node does two real things: (1) it's the actual point where we
    # decide to pause and show the user the collected trip details, and
    # (2) it interprets their answer to either continue or loop back with
    # a correction. Nothing here is a bare if-condition passthrough.
    async def confirm_trip_node(state: GraphState) -> dict:
        answer = interrupt(
            {
                "type": "confirm_trip_details",
                "message": "Here's what I've got for your trip -- confirm to proceed, or tell me what to change.",
                "trip_state": state["trip_state"],
            }
        )
        if isinstance(answer, dict) and answer.get("confirmed") is False and answer.get("message"):
            return {"last_user_message": answer["message"], "ready": False}
        return {}

    async def load_and_route_node(state: GraphState) -> dict:
        intent = state.get("intent")
        if intent == "list_trips":
            trips = await list_trips_for_user(db, state["user_id"])
            return {"assistant_reply": format_trip_list(trips)}
        if intent == "query_other_trip":
            hint = state.get("destination_hint") or ""
            trip = await find_trip_by_destination(db, state["user_id"], hint)
            if trip is None:
                return {"assistant_reply": f"I couldn't find a trip matching '{hint}' in your history."}
            answer = await run_retrieve(state["last_user_message"], trip["itinerary"])
            return {"assistant_reply": answer}
        itinerary = await get_itinerary(db, state["session_id"], state["user_id"])
        answer = await run_retrieve(state["last_user_message"], itinerary)
        return {"assistant_reply": answer}

    async def fetch_and_generate_node(state: GraphState) -> dict:
        fetched = await run_fetch_data(state["trip_state"])
        if fetched.get("fetch_error"):
            return {
                "assistant_reply": fetched["fetch_error"],
                "is_valid": False,
                "validation_issues": [fetched["fetch_error"]],
                "retry_count": MAX_RETRIES,
            }
        trip_state = dict(state["trip_state"])
        if state.get("validation_issues"):
            trip_state["previous_issues_to_fix"] = state["validation_issues"]
        try:
            itinerary = await run_generate(trip_state, fetched["hotels"], fetched["weather"])
        except Exception as e:
            return {
                "assistant_reply": f"Sorry, I had trouble generating the itinerary ({type(e).__name__}). Please try again.",
                "is_valid": False,
                "validation_issues": [f"Generation error: {e}"],
                "retry_count": MAX_RETRIES,
            }
        itinerary_dict = itinerary.model_dump()

        # --- Deterministic day-swap: the LLM regenerates the whole itinerary
        # every time, so a "swap day X and day Y" instruction can't be trusted
        # to survive a full regen. Apply it here as an exact object swap
        # instead, then consume the note so it doesn't reapply on later turns.
        notes = trip_state.get("notes") or []
        remaining_notes = []
        for note in notes:
            if isinstance(note, str) and note.startswith("SWAP_DAYS:"):
                import re
                m = re.search(r"day\s*(\d+)\s*and\s*day\s*(\d+)", note, re.IGNORECASE)
                if m:
                    day_a, day_b = int(m.group(1)), int(m.group(2))
                    days = itinerary_dict.get("days", [])
                    idx_a = next((i for i, d in enumerate(days) if d.get("day_number") == day_a), None)
                    idx_b = next((i for i, d in enumerate(days) if d.get("day_number") == day_b), None)
                    if idx_a is not None and idx_b is not None:
                        days[idx_a], days[idx_b] = days[idx_b], days[idx_a]
                        days[idx_a]["day_number"] = day_a
                        days[idx_b]["day_number"] = day_b
                        # dates stay tied to their calendar position, not to
                        # the content, so keep original dates in place
                # consumed -- don't keep reapplying it on every future regen
                continue
            remaining_notes.append(note)
        if len(remaining_notes) != len(notes):
            trip_state["notes"] = remaining_notes
            state["trip_state"]["notes"] = remaining_notes

        return {
            "hotels": fetched["hotels"],
            "weather": fetched["weather"],
            "itinerary": itinerary_dict,
            "trip_state": state["trip_state"],
        }

    async def validate_and_respond_node(state: GraphState) -> dict:
        if "itinerary" not in state:
            return {"is_valid": False, "validation_issues": state.get("validation_issues", ["No itinerary available"])}

        result = await run_validate(state["trip_state"], state["itinerary"], state["weather"])
        print(f"[VALIDATE] attempt={state.get('retry_count', 0) + 1} is_valid={result['is_valid']} issues={result['issues']}")

        retry_count = state.get("retry_count", 0)
        is_valid = result["is_valid"]

        updates = {
            "is_valid": is_valid,
            "validation_issues": result["issues"],
            "retry_count": retry_count + 1,
        }

        # The budget-floor warning is never blocking (no regenerate can fix a
        # budget that's inherently below the realistic minimum) -- bake it
        # straight into the itinerary's notes so the user sees it immediately
        # instead of getting stuck in a retry/confirm loop that can't resolve.
        if result.get("budget_warning"):
            itinerary = dict(state["itinerary"])
            existing_notes = itinerary.get("notes") or ""
            itinerary["notes"] = (existing_notes + " " if existing_notes else "") + result["budget_warning"]
            updates["itinerary"] = itinerary

        return updates

    # Same principle: this node does the real work of deciding what to save
    # and how to respond, using the user's answer -- not a passthrough flag.
    async def confirm_issues_node(state: GraphState) -> dict:
        itinerary = state.get("itinerary", {})
        answer = interrupt(
            {
                "type": "confirm_generation_issues",
                "message": "I couldn't produce a perfect itinerary. Here's the best attempt and what's wrong with it -- proceed anyway, or tell me what to change?",
                "itinerary": itinerary,
                "issues": state.get("validation_issues", []),
            }
        )
        proceed = isinstance(answer, dict) and answer.get("proceed") is True
        if proceed:
            trip_state = state["trip_state"]
            summary = (
                f"{trip_state.get('destination')} trip ({trip_state.get('start_date')} to "
                f"{trip_state.get('end_date')}), stayed at {itinerary.get('selected_hotel')}, "
                f"~Rs.{itinerary.get('estimated_total_cost')} for {trip_state.get('members')} people. "
                f"(Accepted with known issues.)"
            )
            await save_trip(
                db, state["session_id"], state["user_id"], trip_state,
                itinerary=itinerary, summary=summary, status="generated_with_issues",
                validation_issues=state.get("validation_issues", []),
            )
            return {
                "assistant_reply": f"Here's your itinerary (accepted with noted issues). Estimated cost: Rs.{itinerary.get('estimated_total_cost')}",
            }
        else:
            new_message = answer.get("message") if isinstance(answer, dict) else None
            return {
                "last_user_message": new_message or state["last_user_message"],
                "retry_count": 0,
                "ready": False,
            }

    async def finalize_success_node(state: GraphState) -> dict:
        itinerary = state.get("itinerary")
        if not itinerary:
            reply = state.get("assistant_reply") or "Sorry, no itinerary could be generated."
            return {"assistant_reply": reply}

        trip_state = state["trip_state"]
        summary = (
            f"{trip_state.get('destination')} trip ({trip_state.get('start_date')} to "
            f"{trip_state.get('end_date')}), stayed at {itinerary.get('selected_hotel')}, "
            f"~Rs.{itinerary.get('estimated_total_cost')} for {trip_state.get('members')} people."
        )
        await save_trip(
            db, state["session_id"], state["user_id"], trip_state,
            itinerary=itinerary, summary=summary, status="generated",
            validation_issues=[],
        )
        return {"assistant_reply": f"Here's your itinerary! Estimated cost: Rs.{itinerary.get('estimated_total_cost')}"}

    def route_after_collect(state: GraphState) -> str:
        intent = state.get("intent")
        if intent in ("new_trip_started", "greeting", "farewell", "off_topic", "destination_info"):
            return "end"
        if intent in ("list_trips", "query_other_trip", "query_existing"):
            return "load_and_route"
        return "confirm_trip" if state.get("ready") else "end"

    def route_after_confirm_trip(state: GraphState) -> str:
        return "fetch_and_generate" if state.get("ready", True) else "collect"

    def route_after_validate(state: GraphState) -> str:
        if state.get("is_valid") and "itinerary" in state:
            return "success"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "confirm_issues" if "itinerary" in state else "end"
        return "retry"

    def route_after_confirm_issues(state: GraphState) -> str:
        return "end" if state.get("assistant_reply") else "collect"

    graph = StateGraph(GraphState)
    graph.add_node("collect", collect_node)
    graph.add_node("confirm_trip", confirm_trip_node)
    graph.add_node("load_and_route", load_and_route_node)
    graph.add_node("fetch_and_generate", fetch_and_generate_node)
    graph.add_node("validate_and_respond", validate_and_respond_node)
    graph.add_node("confirm_issues", confirm_issues_node)
    graph.add_node("finalize_success", finalize_success_node)

    graph.set_entry_point("collect")
    graph.add_conditional_edges(
        "collect", route_after_collect,
        {"confirm_trip": "confirm_trip", "load_and_route": "load_and_route", "end": END},
    )
    graph.add_conditional_edges(
        "confirm_trip", route_after_confirm_trip,
        {"fetch_and_generate": "fetch_and_generate", "collect": "collect"},
    )
    graph.add_edge("load_and_route", END)
    graph.add_edge("fetch_and_generate", "validate_and_respond")
    graph.add_conditional_edges(
        "validate_and_respond", route_after_validate,
        {"retry": "fetch_and_generate", "success": "finalize_success", "confirm_issues": "confirm_issues", "end": END},
    )
    graph.add_conditional_edges(
        "confirm_issues", route_after_confirm_issues,
        {"end": END, "collect": "collect"},
    )
    graph.add_edge("finalize_success", END)

    return graph.compile(checkpointer=_checkpointer)
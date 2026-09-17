import uuid
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
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
        trip_state, status = await load_trip(db, state["session_id"])
        result = await run_collect(trip_state, state["last_user_message"])

        # Guardrail: if no itinerary has been generated yet for this session,
        # it makes no sense to classify the message as a query about an
        # existing itinerary -- there's nothing to query. Treat it as
        # continuing to provide/confirm trip details instead.
        if result.intent in ("query_existing",) and status != "generated":
            result.intent = "new_or_modify"

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
                else "Got all the details I need. Let me fetch some options for you..."
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

        itinerary = await get_itinerary(db, state["session_id"])
        answer = await run_retrieve(state["last_user_message"], itinerary)
        return {"assistant_reply": answer}

    async def fetch_and_generate_node(state: GraphState) -> dict:
        fetched = await run_fetch_data(state["trip_state"])
        if fetched.get("fetch_error"):
            return {"assistant_reply": fetched["fetch_error"], "is_valid": True}

        trip_state = dict(state["trip_state"])
        if state.get("validation_issues"):
            trip_state["previous_issues_to_fix"] = state["validation_issues"]

        itinerary = await run_generate(trip_state, fetched["hotels"], fetched["weather"])
        return {
            "hotels": fetched["hotels"],
            "weather": fetched["weather"],
            "itinerary": itinerary.model_dump(),
        }

    async def validate_and_respond_node(state: GraphState) -> dict:
        result = await run_validate(state["trip_state"], state["itinerary"], state["weather"])
        retry_count = state.get("retry_count", 0)
        is_valid = result["is_valid"]
        give_up = retry_count + 1 >= MAX_RETRIES

        reply = ""
        if is_valid or give_up:
            itinerary = state["itinerary"]
            trip_state = state["trip_state"]
            summary = (
                f"{trip_state.get('destination')} trip ({trip_state.get('start_date')} to "
                f"{trip_state.get('end_date')}), stayed at {itinerary.get('selected_hotel')}, "
                f"~Rs.{itinerary.get('estimated_total_cost')} for {trip_state.get('members')} people."
            )
            await save_trip(
                db,
                state["session_id"],
                state["user_id"],
                trip_state,
                itinerary=itinerary,
                summary=summary,
                status="generated" if is_valid else "generated_with_issues",
            )
            reply = f"Here's your itinerary! Estimated cost: {itinerary.get('estimated_total_cost')}"

        return {
            "is_valid": is_valid,
            "validation_issues": result["issues"],
            "retry_count": retry_count + 1,
            "assistant_reply": reply,
        }

    def route_after_collect(state: GraphState) -> str:
        intent = state.get("intent")
        if intent == "new_trip_started":
            return "end"
        if intent in ("list_trips", "query_other_trip", "query_existing"):
            return "load_and_route"
        return "fetch_and_generate" if state.get("ready") else "end"

    def route_after_validate(state: GraphState) -> str:
        if state.get("is_valid"):
            return "done"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "done"
        return "retry"

    graph = StateGraph(GraphState)
    graph.add_node("collect", collect_node)
    graph.add_node("load_and_route", load_and_route_node)
    graph.add_node("fetch_and_generate", fetch_and_generate_node)
    graph.add_node("validate_and_respond", validate_and_respond_node)

    graph.set_entry_point("collect")
    graph.add_conditional_edges(
        "collect",
        route_after_collect,
        {"load_and_route": "load_and_route", "fetch_and_generate": "fetch_and_generate", "end": END},
    )
    graph.add_edge("load_and_route", END)
    graph.add_edge("fetch_and_generate", "validate_and_respond")
    graph.add_conditional_edges(
        "validate_and_respond", route_after_validate, {"retry": "fetch_and_generate", "done": END}
    )

    return graph.compile()
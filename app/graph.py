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
    destination_coordinates: dict
    itinerary: dict
    validation_issues: list
    is_valid: bool
    retry_count: int
    new_session_id: Optional[str]  # returned to client when a new trip thread is auto-started


def build_graph(db: AsyncSession):

    async def load_history_node(state: GraphState) -> dict:
        trip_state, status = await load_trip(db, state["session_id"])
        return {"trip_state": trip_state, "trip_status": status}

    async def collect_node(state: GraphState) -> dict:
        result = await run_collect(state.get("trip_state", {}), state["last_user_message"])
        updated_trip_state = result.updated_trip_state.model_dump()

        # --- New-trip detection ---
        # Current session already has a GENERATED trip, and the user just gave
        # a different destination -> don't overwrite it. Suggest a fresh thread.
        current_status = state.get("trip_status")
        current_destination = (state.get("trip_state") or {}).get("destination")
        new_destination = updated_trip_state.get("destination")

        if (
            result.intent == "new_or_modify"
            and current_status == "generated"
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
            "intent": result.intent,
            "destination_hint": result.destination_hint,
            "missing_fields": result.missing_fields,
            "assistant_reply": reply,
            "ready": not result.missing_fields,
        }

    async def retrieve_node(state: GraphState) -> dict:
        itinerary = await get_itinerary(db, state["session_id"])
        answer = await run_retrieve(state["last_user_message"], itinerary)
        return {"assistant_reply": answer}

    async def retrieve_other_trip_node(state: GraphState) -> dict:
        hint = state.get("destination_hint") or ""
        trip = await find_trip_by_destination(db, state["user_id"], hint)
        if trip is None:
            return {"assistant_reply": f"I couldn't find a trip matching '{hint}' in your history."}
        answer = await run_retrieve(state["last_user_message"], trip["itinerary"])
        return {"assistant_reply": answer}

    async def list_trips_node(state: GraphState) -> dict:
        trips = await list_trips_for_user(db, state["user_id"])
        return {"assistant_reply": format_trip_list(trips)}

    async def fetch_data_node(state: GraphState) -> dict:
        result = await run_fetch_data(state["trip_state"])
        return result

    async def generate_node(state: GraphState) -> dict:
        trip_state = dict(state["trip_state"])
        if state.get("validation_issues"):
            trip_state["previous_issues_to_fix"] = state["validation_issues"]

        itinerary = await run_generate(trip_state, state["hotels"], state["weather"])
        return {"itinerary": itinerary.model_dump()}

    async def validate_node(state: GraphState) -> dict:
        result = await run_validate(state["trip_state"], state["itinerary"], state["weather"])
        retry_count = state.get("retry_count", 0)
        is_valid = result["is_valid"]

        if is_valid or retry_count + 1 >= MAX_RETRIES:
            itinerary = state["itinerary"]
            trip_state = state["trip_state"]
            summary = (
                f"{trip_state.get('destination')} trip ({trip_state.get('start_date')} to "
                f"{trip_state.get('end_date')}), stayed at {itinerary.get('selected_hotel')}, "
                f"~₹{itinerary.get('estimated_total_cost')} for {trip_state.get('members')} people."
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

        return {
            "is_valid": is_valid,
            "validation_issues": result["issues"],
            "retry_count": retry_count + 1,
        }

    def route_after_collect(state: GraphState) -> str:
        intent = state.get("intent")
        if intent == "new_trip_started":
            return "end"
        if intent == "list_trips":
            return "list_trips"
        if intent == "query_other_trip":
            return "retrieve_other"
        if intent == "query_existing":
            return "retrieve"
        return "fetch_data" if state.get("ready") else "end"

    def route_after_validate(state: GraphState) -> str:
        if state.get("is_valid"):
            return "done"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "done"
        return "retry"

    graph = StateGraph(GraphState)
    graph.add_node("load_history", load_history_node)
    graph.add_node("collect", collect_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("retrieve_other", retrieve_other_trip_node)
    graph.add_node("list_trips", list_trips_node)
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("load_history")
    graph.add_edge("load_history", "collect")
    graph.add_conditional_edges(
        "collect",
        route_after_collect,
        {
            "retrieve": "retrieve",
            "retrieve_other": "retrieve_other",
            "list_trips": "list_trips",
            "fetch_data": "fetch_data",
            "end": END,
        },
    )
    graph.add_edge("retrieve", END)
    graph.add_edge("retrieve_other", END)
    graph.add_edge("list_trips", END)
    graph.add_edge("fetch_data", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", route_after_validate, {"retry": "generate", "done": END})

    return graph.compile()
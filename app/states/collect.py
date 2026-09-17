from typing import Optional, Literal
from pydantic import BaseModel, Field
from app.llm import call_structured

REQUIRED_FIELDS = ["destination", "start_date", "end_date", "members"]

SYSTEM_PROMPT = """You are the intent + extraction step of a travel planning agent.

Classify the user's intent into exactly one of:
- "new_or_modify": user is providing trip details for the CURRENT trip being planned, or wants changes to it.
- "query_existing": user is asking about the CURRENT trip's already-generated itinerary (e.g. "what's my hotel", "show me day 2").
- "query_other_trip": user is asking about a DIFFERENT, previously planned trip by name (e.g. "what was my hotel in Goa", "what did I plan for Manali").
- "list_trips": user is asking to see all their trips (e.g. "what trips do I have", "show my trip history").

If intent is "new_or_modify", also update the trip state with any new details
mentioned in the latest user message. Do not invent values, only fill fields
the user has actually specified.
Required fields: destination, start_date, end_date, members.
Budget is optional — if the user doesn't mention it, don't ask for it.
If any required fields are missing, write one natural, friendly question in
clarifying_question asking specifically for those missing fields. Otherwise
leave clarifying_question empty.

For "query_existing", "query_other_trip", "list_trips": leave updated_trip_state
unchanged (echo the current state back) and leave clarifying_question empty.
If intent is "query_other_trip", extract the destination name they're referring
to into destination_hint.
"""


class TripDetails(BaseModel):
    destination: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    members: Optional[int] = None
    budget: Optional[float] = None
    interests: Optional[list[str]] = Field(default=None)


class CollectResult(BaseModel):
    intent: Literal["new_or_modify", "query_existing", "query_other_trip", "list_trips"]
    updated_trip_state: TripDetails
    missing_fields: list[str] = Field(default_factory=list)
    clarifying_question: Optional[str] = None
    destination_hint: Optional[str] = None


async def run_collect(trip_state: dict, user_message: str) -> CollectResult:
    user_content = f"Current trip state:\n{trip_state}\n\nLatest user message:\n{user_message}"
    result = await call_structured(SYSTEM_PROMPT, user_content, CollectResult)

    if result.updated_trip_state.interests is None:
        result.updated_trip_state.interests = []

    if result.intent == "new_or_modify":
        merged = result.updated_trip_state.model_dump()
        actual_missing = [f for f in REQUIRED_FIELDS if not merged.get(f)]
        result.missing_fields = actual_missing
        if not actual_missing:
            result.clarifying_question = None
    else:
        result.missing_fields = []
        result.clarifying_question = None

    return result
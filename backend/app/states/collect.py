import re
from datetime import date
from typing import Optional, Literal
from pydantic import BaseModel, Field
from app.llm import call_structured

REQUIRED_FIELDS = ["destination", "start_date", "end_date", "members"]
MAX_TRIP_DAYS = 21
ONE_DAY_NOTE = "one-day trip"

FIELD_LABELS = {
    "destination": "destination",
    "start_date": "start date",
    "end_date": "end date",
    "members": "number of travellers",
}

# "one day", "1 day", "a day trip", "day trip", "same day", "single day"
_ONE_DAY_RE = re.compile(
    r"\b(one|1|a|single)[\s-]*day\b|\bday[\s-]*trip\b|\bsame[\s-]*day\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are the intent + extraction step of a travel planning agent.

Classify the user's intent into exactly one of:
- "new_or_modify": user is providing trip details for the CURRENT trip being planned, or wants changes to it, or explicitly asks you to plan/build a full itinerary.
- "query_existing": user is asking about the CURRENT trip's already-generated itinerary (e.g. "what's my hotel", "show me day 2").
- "query_other_trip": user is asking about a DIFFERENT, previously planned trip by name, OR asking to see/show a specific already-generated itinerary by destination name (e.g. "what was my hotel in Goa", "what did I plan for Manali", "show me the Ooty itinerary", "show my Vagamon plan").
- "list_trips": user is asking to see all their trips (e.g. "what trips do I have", "show my trip history").
- "greeting": user is just greeting (hi, hello, hey, vanakkam) with no trip details or travel question yet.
- "farewell": user is saying bye/goodbye/thanks-and-closing (bye, goodbye, see you, thanks bye, vanakkam bye) with no further trip request.
- "destination_info": user is asking what places/attractions to see in a destination, WITHOUT asking you to plan a full trip yet (e.g. "places to visit in Pondy", "what can I see in Ooty"). Only classify as new_or_modify if they explicitly ask to plan/build an itinerary.
  FOLLOW-UPS: if the user is continuing a destination_info conversation -- asking
  for "more", "other places", "something different", "some more other than this"
  etc. -- classify this as "destination_info" too, using the destination already
  present in the current trip state (do NOT classify these as off_topic just
  because the message alone doesn't repeat the destination name or the word
  "places").
- "off_topic": the message is unrelated to trip planning entirely (e.g. general knowledge questions, small talk with no travel intent, sports scores, technical/coding questions, or any follow-up question continuing an off-topic thread).

If intent is "new_or_modify", also update the trip state with any new details
mentioned in the latest user message. Do not invent values, only fill fields
the user has actually specified.
Required fields: destination, start_date, end_date, members.
Dates must be extracted in YYYY-MM-DD format. members must be a positive
whole number of travelers (never 0 or negative).
Budget is optional -- if the user doesn't mention it, don't ask for it.
NEVER set previous_budget yourself -- always leave it null. It is managed
entirely by code after your response, not by you.
Any specific instructions about the itinerary itself (e.g. "skip Old Goa",
"keep day 3 in South Goa only", "no water sports", "vegetarian food only")
must be added as short strings to the notes list. Keep existing notes and
append new ones; remove a note only if the user withdraws it.

FLIGHT AND CAR RENTAL -- OPTIONAL, USER-STATED ONLY:
- If the user mentions their own flight details (airline, flight number,
  departure/arrival times, airports/cities, gate), extract them into the
  `flight` fields. If they mention car rental details (pickup/drop-off date,
  time, location, rental company), extract them into the `car_rental`
  fields.
- These are NEVER required and you must NEVER ask for them, invent them, or
  fill in a field the user didn't actually state. Only populate a sub-field
  when the user's message contains that specific piece of information.
  Leave `flight`/`car_rental` entirely null if the user hasn't mentioned
  either.
- If the user updates or corrects a previously given flight/car rental
  detail, update just that field and keep the rest as previously given.
If any required fields are missing, write one natural, friendly question in
clarifying_question asking specifically for those missing fields. Otherwise
leave clarifying_question empty.

DATE RULES:
- If the user says "one day", "1 day", "a day", "day trip" or "same day",
  set end_date EQUAL to start_date and do NOT ask for end_date. Also add the
  note "one-day trip" to notes (if not already there).
- If the user gives a duration in days (e.g. "3 days") and a start_date,
  compute end_date = start_date + (days - 1). Example: "3 days" starting
  2026-09-21 means end_date is 2026-09-23, not 2026-09-24.
- Resolve "today", "tomorrow", "this weekend", "next Friday" etc. using the
  today's date given in the prompt. Never ask for a date the user has
  already implied.
- Accept any date format the user types (e.g. 21/Sep/2026, 21-09-2026) and
  always output YYYY-MM-DD.
- Only ask about fields that are truly missing.

Destination normalization: always expand common short forms, nicknames, and
misspellings to the standard place name (e.g. "pody"/"pondy" -> "Puducherry",
"blr" -> "Bangalore", "madras" -> "Chennai", "bombay" -> "Mumbai").

Source vs destination: if the user describes travel FROM one place TO another
(e.g. "Chennai to Pollachi", "from Delhi to Manali"), the trip destination is
ONLY the place after "to" -- the place BEFORE "to" is just their starting
point, not part of the destination. Never combine both into destination.

For "greeting": write a short, warm one-line reply introducing yourself as a
trip planning assistant and asking where they'd like to go, into direct_reply.
For "farewell": write a short one-line reply wishing them a good trip, into
direct_reply.
For "destination_info": write a short list (4-6 items) of notable
places/things to do in the mentioned destination into direct_reply, then ask
if they'd like a full itinerary planned for it. Use REAL, specific named
places for that destination -- never generic placeholders like "Spot A" or
"Place 1". Also set updated_trip_state.destination to the (normalized)
destination being discussed, even though this isn't a full trip yet, so a
follow-up like "show me more" still knows the destination. Check
destination_info_shown in the current trip state first -- if it already
lists places you mentioned before, do NOT repeat them; list genuinely
different real places instead. Then set updated_trip_state.destination_info_shown
to the full list of places you have now shown across this conversation
(previous ones plus any new ones from this reply).
For "off_topic": write a brief, polite reply into direct_reply saying you're
a trip planning assistant so you can't help with that, and ask if they'd
like to plan a trip instead. Never attempt to actually answer the off-topic
question.
For "list_trips", "query_existing", "query_other_trip": leave direct_reply empty.

For "query_existing", "query_other_trip", "list_trips", "greeting", "farewell",
"off_topic": leave updated_trip_state unchanged (echo the current state back)
and leave clarifying_question empty.
If intent is "query_other_trip", extract the destination name they're referring
to into destination_hint.
REPLY STYLE:
- Keep every direct_reply to 1-2 short sentences.
- For "destination_info": list max 4 places, one short line each, then ask
  in one short line if they want a full itinerary.
- For "greeting": one short line only.
- For "farewell": one short line only.
- No emojis, no long introductions.
"""


class FlightDetails(BaseModel):
    date: Optional[str] = Field(default=None, description="Flight date, YYYY-MM-DD if stated")
    departs_time: Optional[str] = None
    airline: Optional[str] = None
    departure_location: Optional[str] = None
    gate: Optional[str] = None
    arrives_time: Optional[str] = None
    arrival_location: Optional[str] = None


class CarRentalDetails(BaseModel):
    pickup_date: Optional[str] = Field(default=None, description="YYYY-MM-DD if stated")
    pickup_time: Optional[str] = None
    pickup_location: Optional[str] = None
    dropoff_date: Optional[str] = Field(default=None, description="YYYY-MM-DD if stated")
    dropoff_time: Optional[str] = None
    dropoff_location: Optional[str] = None
    company: Optional[str] = None


class TripDetails(BaseModel):
    destination: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    members: Optional[int] = None
    budget: Optional[float] = None
    interests: Optional[list[str]] = Field(default=None)
    notes: Optional[list[str]] = Field(default=None)
    previous_budget: Optional[float] = Field(
        default=None,
        description="Internal only -- set when the user reduces the budget, so generate.py can explain what got cut.",
    )
    destination_info_shown: Optional[list[str]] = Field(
        default=None,
        description="Places already listed to the user during destination_info replies, so follow-ups don't repeat them.",
    )
    flight: Optional[FlightDetails] = Field(
        default=None, description="Only populated from what the user explicitly states -- never invented."
    )
    car_rental: Optional[CarRentalDetails] = Field(
        default=None, description="Only populated from what the user explicitly states -- never invented."
    )


class CollectResult(BaseModel):
    intent: Literal[
        "new_or_modify", "query_existing", "query_other_trip", "list_trips",
        "greeting", "farewell", "destination_info", "off_topic"
    ]
    updated_trip_state: TripDetails
    missing_fields: list[str] = Field(default_factory=list)
    clarifying_question: Optional[str] = None
    destination_hint: Optional[str] = None
    direct_reply: Optional[str] = None


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "n/a"):
        return True
    return False


def _parse_date(value: str) -> Optional[date]:
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _is_one_day_trip(user_message: str, merged: dict) -> bool:
    """True if the latest message or any saved note says it's a one-day trip."""
    if _ONE_DAY_RE.search(user_message or ""):
        return True
    for note in merged.get("notes") or []:
        if _ONE_DAY_RE.search(str(note)):
            return True
    return False


def _validate_structure(merged: dict) -> tuple[list[str], Optional[str], dict]:
    """
    Validates members/budget/date values beyond simple presence checks.
    Returns (invalid_field_names, error_message, corrected_merged_dict).
    """
    invalid = []
    messages = []

    # --- members must be a positive integer ---
    members = merged.get("members")
    if members is not None and not _is_missing(members):
        try:
            if int(members) <= 0:
                invalid.append("members")
                messages.append("the number of travelers must be at least 1")
                merged["members"] = None
        except (TypeError, ValueError):
            invalid.append("members")
            messages.append("the number of travelers should be a whole number")
            merged["members"] = None

    # --- budget, if given, must be positive ---
    budget = merged.get("budget")
    if budget is not None and not _is_missing(budget):
        try:
            if float(budget) <= 0:
                merged["budget"] = None
                messages.append("budget should be a positive amount, so I've left it unset")
        except (TypeError, ValueError):
            merged["budget"] = None
            messages.append("budget should be a number, so I've left it unset")

    # --- dates must parse, and start must be <= end ---
    start_raw, end_raw = merged.get("start_date"), merged.get("end_date")
    start = _parse_date(start_raw) if not _is_missing(start_raw) else None
    end = _parse_date(end_raw) if not _is_missing(end_raw) else None

    if start_raw and not _is_missing(start_raw) and start is None:
        invalid.append("start_date")
        messages.append("the start date wasn't in a recognizable format (use YYYY-MM-DD)")
        merged["start_date"] = None
        start = None

    if end_raw and not _is_missing(end_raw) and end is None:
        invalid.append("end_date")
        messages.append("the end date wasn't in a recognizable format (use YYYY-MM-DD)")
        merged["end_date"] = None
        end = None

    if start and end:
        if start > end:
            merged["start_date"], merged["end_date"] = end.isoformat(), start.isoformat()
            start, end = end, start
            messages.append(
                f"your dates looked swapped, so I've set the trip as "
                f"{merged['start_date']} to {merged['end_date']}"
            )

        trip_days = (end - start).days + 1
        if trip_days > MAX_TRIP_DAYS:
            invalid.extend(["start_date", "end_date"])
            merged["start_date"] = None
            merged["end_date"] = None
            messages.append(
                f"trips longer than {MAX_TRIP_DAYS} days aren't supported yet -- "
                f"could you pick a shorter date range?"
            )
        elif start < date.today():
            invalid.extend(["start_date", "end_date"])
            merged["start_date"] = None
            merged["end_date"] = None
            messages.append(
                f"{start.isoformat()} is in the past -- could you share a future date instead?"
            )

    combined_message = "; ".join(messages) if messages else None
    return invalid, combined_message, merged


async def run_collect(trip_state: dict, user_message: str, trip_status: str = "new") -> CollectResult:
    """
    trip_status: status of the currently loaded trip ('new', 'collecting',
    'generated', etc). Used to correct the LLM's intent BEFORE computing
    missing_fields.
    """
    today = date.today()
    user_content = (
        f"Today's date is {today.isoformat()} ({today.strftime('%A')}). Use this to resolve "
        f"any relative dates the user gives (e.g. 'this Saturday', 'next week', 'in 5 days', "
        f"'tomorrow') into exact YYYY-MM-DD dates.\n\n"
        f"Current trip state:\n{trip_state}\n\nLatest user message:\n{user_message}"
    )
    result = await call_structured(SYSTEM_PROMPT, user_content, CollectResult)

    if result.updated_trip_state.interests is None:
        result.updated_trip_state.interests = []
    if result.updated_trip_state.notes is None:
        result.updated_trip_state.notes = []

    # Guardrail: correct intent BEFORE missing-field computation.
    if result.intent == "query_existing" and trip_status != "generated":
        result.intent = "new_or_modify"

    if result.intent == "new_or_modify":
        merged = result.updated_trip_state.model_dump()

        # --- Track budget reductions so generate.py can explain what was cut ---
        old_budget = trip_state.get("budget")
        new_budget = merged.get("budget")
        if (
            old_budget is not None
            and new_budget is not None
            and not _is_missing(new_budget)
            and float(new_budget) < float(old_budget)
        ):
            merged["previous_budget"] = old_budget
        elif "previous_budget" in trip_state:
            # Carry forward an existing previous_budget only if budget wasn't
            # touched this turn; otherwise drop it so a non-reduction (or an
            # increase) doesn't keep triggering the "budget cut" explanation.
            if new_budget is None or _is_missing(new_budget) or float(new_budget) >= float(old_budget or 0):
                merged.pop("previous_budget", None)

        # --- One-day trip handling ---
        one_day = _is_one_day_trip(user_message, merged)
        if one_day:
            # Remember it in notes so later turns ("tomorrow, 2 people") still know.
            notes = merged.get("notes") or []
            if ONE_DAY_NOTE not in notes:
                merged["notes"] = notes + [ONE_DAY_NOTE]
            # end_date always equals start_date for a one-day trip.
            if not _is_missing(merged.get("start_date")):
                merged["end_date"] = merged["start_date"]

        invalid_fields, validation_message, merged = _validate_structure(merged)
        result.updated_trip_state = TripDetails(**merged)

        actual_missing = [f for f in REQUIRED_FIELDS if _is_missing(merged.get(f))]
        actual_missing = list(dict.fromkeys(actual_missing + [f for f in invalid_fields if f in REQUIRED_FIELDS]))

        # One-day trip: never ask for end_date (it equals start_date).
        if one_day:
            actual_missing = [f for f in actual_missing if f != "end_date"]

        result.missing_fields = actual_missing

        if actual_missing:
            labels = [FIELD_LABELS.get(f, f) for f in actual_missing]
            base_question = f"Could you tell me the trip's {', '.join(labels)}?"
            result.clarifying_question = (
                f"{validation_message}. {base_question}" if validation_message else base_question
            )
        elif validation_message:
            result.clarifying_question = validation_message
            result.missing_fields = []
        else:
            result.clarifying_question = None
    else:
        result.missing_fields = []
        result.clarifying_question = None

        if result.intent == "destination_info":
            merged = result.updated_trip_state.model_dump()

            # Carry forward destination if the LLM didn't restate it on a
            # short follow-up ("show me more") -- context comes from trip_state.
            if _is_missing(merged.get("destination")) and not _is_missing(trip_state.get("destination")):
                merged["destination"] = trip_state.get("destination")

            # Merge previously shown places with any new ones, de-duplicated,
            # so repeats don't creep back in even if the LLM's own list slips.
            existing_shown = trip_state.get("destination_info_shown") or []
            llm_shown = merged.get("destination_info_shown") or []
            merged["destination_info_shown"] = list(dict.fromkeys(existing_shown + llm_shown))

            result.updated_trip_state = TripDetails(**merged)

    return result
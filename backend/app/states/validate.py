from app.llm import call_structured
from pydantic import BaseModel, Field

# Rough real-world floors: even a very frugal budget trip realistically costs
# at least this much per person per day (basic stay + food + local transport).
# These are deliberately conservative/low so they only catch truly impossible
# budgets, not just "tight" ones -- the LLM handles nuance beyond this.
# Destinations not listed fall back to DEFAULT_MIN_COST_PER_PERSON_PER_DAY.
# NOTE: min_budget_per_day (from live web search) takes priority over this
# dictionary when available -- see run_validate below.
DESTINATION_MIN_COST_PER_PERSON_PER_DAY = {
    "goa": 900,
    "kerala": 900,
    "kashmir": 1000,
    "ladakh": 1200,
    "manali": 900,
    "shimla": 900,
    "ooty": 800,
    "munnar": 850,
    "andaman": 1300,
    "rajasthan": 900,
}
DEFAULT_MIN_COST_PER_PERSON_PER_DAY = 800

# How much over the stated budget the generated itinerary is allowed to be
# before we treat it as a hard validation failure and force a regenerate.
BUDGET_OVERSHOOT_TOLERANCE = 1.1  # 10%

SYSTEM_PROMPT = """You are the validation step of a travel planning agent.
Given the trip details, the generated itinerary, and the weather forecast,
check the itinerary for problems.

Check for:
- Activities that don't fit the destination or are unrealistic.
- Estimated total cost significantly exceeding the stated budget.
- Outdoor activities scheduled on days with high rain probability without a warning.
- Hotel name not matching any of the hotels actually offered, or the
  itinerary mentioning MORE THAN ONE hotel when only one was selected.
  EXCEPTION: if the selected hotel's name contains "(no listings found)",
  this is an intentional system-generated placeholder used when no real
  hotel listings were available nearby -- it IS the only option that was
  offered, not a fabricated one. Do NOT flag this as an issue.
- Missing or incomplete day plans (wrong number of days vs trip duration).
- Unrealistic travel logistics: if the itinerary involves driving between
  multiple cities/towns, check that travel times are reasonable and that
  the same day doesn't require impossible back-and-forth trips.

Do NOT flag the stated budget itself as too low/unrealistic -- that check is
already handled separately and deterministically, and is never something a
regenerate can fix. Only flag budget problems that regeneration CAN actually
fix, e.g. the itinerary's own estimated cost overshooting an achievable
budget through avoidable choices.

IMPORTANT: The weather data provided may be historical data from a PAST
YEAR used only as a typical-conditions estimate for a far-future trip (this
is intentional and clearly labeled). Do NOT flag a year mismatch between
the weather dates and the trip dates as an issue -- only evaluate whether
the temperature/rain patterns shown are being used sensibly (e.g. suggesting
indoor activities on days with high rain probability). If rain probability
data is null/missing, do not flag that as an issue either -- just skip
weather-based checks for those days.

If everything looks fine, return is_valid=True with an empty issues list.
Otherwise return is_valid=False and a list of specific, actionable issues
so the itinerary can be regenerated to fix them.
"""


class ValidationResult(BaseModel):
    is_valid: bool
    issues: list[str] = Field(default_factory=list)


def _get_min_cost_per_day(destination: str) -> int:
    """
    Looks up a destination-aware minimum realistic cost per person per day.
    Falls back to a conservative default if the destination isn't in the table.
    Matching is loose (substring, case-insensitive) so "North Goa" or
    "Goa, India" still match "goa".
    """
    if not destination:
        return DEFAULT_MIN_COST_PER_PERSON_PER_DAY

    destination_lower = destination.lower()
    for key, min_cost in DESTINATION_MIN_COST_PER_PERSON_PER_DAY.items():
        if key in destination_lower:
            return min_cost

    return DEFAULT_MIN_COST_PER_PERSON_PER_DAY


def _hard_validate(trip_state: dict, itinerary: dict, min_budget_per_day: int = None) -> tuple[list[str], str | None]:
    """
    Deterministic checks that don't rely on LLM judgment -- catches things
    an LLM can be talked into rationalizing away, like impossible budgets.

    Returns (blocking_issues, budget_warning):
    - blocking_issues: things regeneration CAN actually fix (e.g. an
      itinerary that overshoots an achievable budget -- pick cheaper
      options). These force a retry/regenerate.
    - budget_warning: set when the STATED budget itself is below the
      realistic floor for this trip. No amount of regenerating changes
      that floor, so this is never treated as a blocking issue -- it's
      surfaced as a note on the itinerary instead, so the user gets their
      plan immediately with an honest caveat rather than being stuck in a
      retry loop that can never succeed.

    min_budget_per_day: an optional live-search-derived minimum (per person,
    per day) for this specific destination. When provided, it takes priority
    over the static dictionary/default below, since it's more accurate and
    covers destinations the dictionary doesn't know about.
    """
    issues = []
    budget_warning = None

    members = trip_state.get("members") or 1
    days = len(itinerary.get("days", []))
    budget = trip_state.get("budget")
    destination = trip_state.get("destination", "")

    if budget and days > 0 and members > 0:
        min_cost_per_day = min_budget_per_day or _get_min_cost_per_day(destination)
        min_realistic_total = min_cost_per_day * members * days
        if budget < min_realistic_total:
            budget_warning = (
                f"Your budget of Rs.{budget} is below the realistic minimum for "
                f"{members} people over {days} days in {destination or 'this destination'} -- "
                f"a real trip would cost at least approximately Rs.{min_realistic_total} even "
                f"at the most frugal level (basic stay + food + local transport, "
                f"~Rs.{min_cost_per_day}/person/day). This plan is built as close to your "
                f"budget as realistically possible, but the true minimum cost is higher."
            )
        else:
            # Budget is realistically achievable, so an itinerary that still
            # overshoots it is a generation failure, not an inherent
            # impossibility -- force a cheaper regenerate instead of letting
            # it slide with an inflated estimated_total_cost.
            estimated_cost = itinerary.get("estimated_total_cost")
            if estimated_cost and estimated_cost > budget * BUDGET_OVERSHOOT_TOLERANCE:
                issues.append(
                    f"The itinerary's estimated cost of Rs.{estimated_cost} exceeds the "
                    f"stated budget of Rs.{budget} by more than "
                    f"{int((BUDGET_OVERSHOOT_TOLERANCE - 1) * 100)}%, even though this budget "
                    f"is realistically achievable for {members} people over {days} days. "
                    f"Regenerate a cheaper plan (cheaper hotel and/or activity choices) that "
                    f"fits within Rs.{budget}."
                )

    return issues, budget_warning


async def run_validate(trip_state: dict, itinerary: dict, weather: dict, min_budget_per_day: int = None) -> dict:
    hard_issues, budget_warning = _hard_validate(trip_state, itinerary, min_budget_per_day)
    if hard_issues:
        return {"is_valid": False, "issues": hard_issues, "budget_warning": budget_warning}

    is_forecast = weather.get("is_forecast", True)
    weather_note = (
        "This is a REAL forecast for the exact trip dates."
        if is_forecast
        else "This is HISTORICAL data from last year, used only as a typical-weather "
        "estimate since the trip is too far in the future for a real forecast. "
        "The year will not match the trip dates -- that is expected, not an error."
    )

    user_content = f"""
Trip details: {trip_state}

Generated itinerary: {itinerary}

Weather data ({weather_note}): {weather.get('daily', {})}

Validate this itinerary.
"""
    result = await call_structured(SYSTEM_PROMPT, user_content, ValidationResult)
    issues = result.issues

    # Deterministic safety net: even if the LLM ignores the prompt
    # instruction above, never let a hotel-mismatch complaint block an
    # itinerary that correctly used the intentional "no listings found"
    # placeholder -- there is no other hotel it could have picked, so
    # regenerating can never resolve this.
    selected_hotel = (itinerary.get("selected_hotel") or "")
    if "no listings found" in selected_hotel.lower():
        issues = [i for i in issues if "hotel" not in i.lower()]

    return {"is_valid": not issues, "issues": issues, "budget_warning": budget_warning}
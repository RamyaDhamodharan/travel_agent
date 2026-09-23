from typing import Optional
from pydantic import BaseModel, Field
from app.llm import call_structured
from app.states.collect import FlightDetails, CarRentalDetails

SYSTEM_PROMPT = """You are the itinerary generation step of a travel planning agent.
Given the trip details, weather forecast, and available hotels, create a
realistic day-by-day itinerary.

Rules:
- ONE-DAY TRIPS: if the trip details say it is a one-day trip (start_date
  equals end_date), NO hotel is needed. Set selected_hotel to exactly
  "N/A (one-day trip, no hotel needed)", do NOT mention any hotel, check-in
  or check-out anywhere, and do NOT include accommodation in the cost.
  Produce exactly ONE day in days, planned from morning to evening
  (breakfast, sightseeing, lunch, evening activity, dinner), with realistic
  travel times between places inside the city.
- For multi-day trips: choose exactly ONE hotel from the provided list as
  selected_hotel, and keep the itinerary consistent with staying at that ONE
  hotel for the whole trip. Do not check in/out of multiple hotels or mention
  a second hotel anywhere in the itinerary. Leave `hotel_options` null --
  that list is filled in separately from the real fetched hotel data, not by
  you.
- If the destination spans multiple towns/cities (e.g. a hill-station
  region), prefer a SINGLE base location for the whole trip and take day
  trips from there, rather than moving hotels each day. Only plan
  multi-city travel if it's genuinely necessary, and if so, keep driving
  times realistic and never schedule a same-day round trip of more than
  ~2-3 hours total driving alongside other activities.
- Keep activities realistic for the number of days and destination.

TIMING AND TRAVEL:
- EVERY activity must have a realistic clock time (e.g. "9:00 AM") in its
  `time` field, in a sensible order across the day (morning to night).
- EVERY activity except the first one of the day must have a short
  `travel_time_from_previous` estimate (e.g. "20 min drive", "10 min walk")
  based on realistic distance between that activity and the one before it.
  The first activity of the day can leave this empty/null.

MEALS -- NO COST, EXACTLY 5 NEARBY OPTIONS PER MEAL WITH DISTANCE + SPECIALTY, NO TIME:
- For each day, populate `meals` with FOUR entries, ALWAYS, in this exact
  order: Breakfast, Lunch, Snacks, Dinner. Snacks is NOT optional -- every
  single day, even short or activity-packed days, MUST include a Snacks
  entry with its own 5 options near the traveller's afternoon/evening
  location that day.
- Each meal needs `options`: a list of EXACTLY 5 DIFFERENT specific real
  eateries/restaurants/street-food stalls (you may include the hotel
  restaurant as one of the 5, but the other 4+ must be distinct real places
  -- never pad by repeating the hotel or reusing the same name twice) NEAR
  wherever the itinerary already has the traveller around that time of day.
  This is a HARD REQUIREMENT: fewer than 5 options for any meal is an
  incomplete, incorrect response. Give the traveller a real choice among
  5 genuinely different places, not 2. Mix well-known restaurants with local
  street-food stalls, cafes, and casual spots so the 5 feel varied, not
  interchangeable.
- EVERY option MUST state its distance/travel time FROM the specific nearby
  place it's close to (e.g. name="Lakeview Café -- seafood",
  distance="5 min walk from Ooty Lake"; name="Hotel restaurant",
  distance="on-site at the hotel"). Never leave distance vague or blank --
  always name the reference place it's measured from.
- EVERY option MUST also state its `specialty`: the specific dish or thing
  it's known for (e.g. "Mysore masala dosa", "butter chicken", "filter
  coffee", "kebabs"). Keep it short -- a few words, not a sentence.
- Do NOT include a clock time for meals -- just the meal name and its
  nearby options with distances and specialties. The day's activity timings
  already anchor when meals roughly happen.
- Do NOT put a cost/price on any meal, and do NOT add meal costs into
  estimated_total_cost or budget_breakdown. Meals are recommendations only.

BUDGET:
- The weather data given may be historical data from last year (used as a
  typical-conditions estimate for far-future trips) rather than an exact
  forecast -- if so, this is expected; just use the temperature/rain
  patterns to decide indoor vs outdoor activities, don't worry about the
  year not matching the trip dates.
- If rain probability data is missing/null, do not make weather-based
  assumptions for those days.
- If the weather data has an "error" or is empty, plan sensibly without
  weather assumptions and mention that weather was unavailable in notes.
- All costs, budgets, and estimated_total_cost must be in Indian Rupees (₹).
- If a budget is stated, stay within it as a rough guide. If no budget is
  stated, suggest a reasonable, moderate-cost plan in Indian Rupees and
  mention the estimated total cost clearly in notes. Do NOT invent or assume
  a specific budget figure if none was given -- just describe the plan as
  moderate-cost.
- ALWAYS populate `budget_breakdown` with your best-estimate split of
  estimated_total_cost across: accommodation, transport (all travel
  including to/from destination and local transport), activities (paid
  entry fees/tours), and miscellaneous (anything else, e.g. shopping
  buffer -- NOT meals, meals are never costed). These four numbers should
  roughly sum to estimated_total_cost.
- BUDGET WAS REDUCED FROM A PREVIOUS ATTEMPT: if the trip details include a
  `previous_budget` that is HIGHER than the current `budget`, you are
  regenerating a cheaper version of the plan. You MUST populate
  `budget_cuts` with a short list of exactly what was downgraded or removed
  to hit the new lower budget compared to a typical plan at the previous
  budget (e.g. "Switched to a 2-star hotel instead of 3-star", "Dropped the
  paid heritage tour on day 2", "Reduced private cab use in favor of
  shared/local transport"). If budget was not reduced, leave budget_cuts
  empty.
- The trip details may include a "notes" list of specific user instructions
  (e.g. "skip Old Goa", "keep day 3 in South Goa only"). You MUST follow
  EVERY entry in that list (the note "one-day trip" just means a single day,
  it is not an activity). Never include a place or activity the user asked
  to skip, and respect any day-specific instructions exactly.
- Each day should have a short list of activities/places, not a wall of text.
- You MUST always call the tool and produce a complete itinerary, even if
  the hotel list looks incomplete or generic. If no real hotel names are
  available, use the first entry from the list as-is and mention in notes
  that hotel options were limited. NEVER refuse to generate an itinerary.
- FLIGHT AND CAR RENTAL: leave the `flight` and `car_rental` fields
  completely null. Never invent flight or car rental details -- these are
  filled in separately from what the user actually stated, not by you.
- Keep it concise: each day's summary max 10 words, 3-4 activities per
  day, each activity's `place` max 12 words. Keep notes to 1-2 short
  sentences.
  """

DAY_TRIP_HOTEL = "N/A (one-day trip, no hotel needed)"


class Activity(BaseModel):
    time: str = Field(description="Clock time, e.g. '9:00 AM'")
    place: str = Field(description="Short activity/place description, max 12 words")
    travel_time_from_previous: Optional[str] = Field(
        default=None,
        description="Estimated travel time from the previous activity, e.g. '20 min drive'. Empty for the first activity of the day.",
    )


class MealOption(BaseModel):
    name: str = Field(description="Eatery/restaurant/street-food stall name or short description, e.g. 'Lakeview Café -- seafood'")
    distance: str = Field(
        description="Distance/travel time FROM the relevant place that day, e.g. '5 min walk from Ooty Lake' or '10 min drive from the hotel'"
    )
    specialty: str = Field(
        description="What it's known for -- a specific dish or item, kept short, e.g. 'Mysore masala dosa', 'butter chicken', 'filter coffee'"
    )


class MealSuggestion(BaseModel):
    meal: str = Field(description="Breakfast, Lunch, Snacks, or Dinner")
    options: list[MealOption] = Field(
        description="EXACTLY 5 different nearby eateries/street-food stalls/hotel restaurant options for this meal, each with its distance from the relevant nearby place and its specialty dish. No cost/price, no time.",
        min_length=5,
        max_length=5,
    )


class DayPlan(BaseModel):
    day_number: int
    date: str
    summary: str = Field(description="One-line summary of the day's theme")
    activities: list[Activity] = Field(description="3-5 timed activities/places for the day")
    meals: list[MealSuggestion] = Field(
        description="Meal recommendations for the day, no cost. MUST contain exactly 4 entries, in order: Breakfast, Lunch, Snacks, Dinner -- Snacks is mandatory every day, never skipped.",
        min_length=4,
        max_length=4,
    )


class BudgetBreakdown(BaseModel):
    accommodation: Optional[float] = None
    transport: Optional[float] = None
    activities: Optional[float] = None
    miscellaneous: Optional[float] = None


class HotelOption(BaseModel):
    name: str
    address: Optional[str] = None
    selected: bool = Field(default=False, description="True for the one hotel this itinerary is actually built around")


class Itinerary(BaseModel):
    destination: str
    selected_hotel: str = Field(
        description="Name of the ONE chosen hotel from the provided list, used for the entire stay. "
        "For one-day trips use 'N/A (one-day trip, no hotel needed)'."
    )
    hotel_options: Optional[list[HotelOption]] = Field(
        default=None,
        description="All hotels that were actually available nearby, not just the selected one. "
        "Always set deterministically from the real fetched hotel list after generation -- never trust the LLM's own value here.",
    )
    days: list[DayPlan]
    estimated_total_cost: float
    budget_breakdown: Optional[BudgetBreakdown] = Field(
        default=None, description="Split of estimated_total_cost across accommodation/transport/activities/misc"
    )
    budget_cuts: Optional[list[str]] = Field(
        default=None, description="What was downgraded/removed if budget was reduced from a previous attempt"
    )
    notes: Optional[str] = Field(default=None, description="Any caveats, e.g. weather warnings")
    flight: Optional[FlightDetails] = Field(
        default=None,
        description="User-stated flight details, carried through as-is. Always overwritten from trip_state after generation -- never trust the LLM's own value here.",
    )
    car_rental: Optional[CarRentalDetails] = Field(
        default=None,
        description="User-stated car rental details, carried through as-is. Always overwritten from trip_state after generation -- never trust the LLM's own value here.",
    )


def _validate_itinerary_meals(result: "Itinerary") -> Optional[str]:
    """Post-hoc check beyond pydantic's own min/max_length, so we get a
    clear, specific correction message to feed back to the model (pydantic's
    own error is often too generic to fix the right day/meal)."""
    for day in result.days:
        meal_names = [m.meal for m in day.meals]
        expected = ["Breakfast", "Lunch", "Snacks", "Dinner"]
        if len(day.meals) != 4 or any(
            m.lower() != e.lower() for m, e in zip(meal_names, expected)
        ):
            return (
                f"Day {day.day_number} must have exactly 4 meals in this order: "
                f"Breakfast, Lunch, Snacks, Dinner. Got: {meal_names}"
            )
        for meal in day.meals:
            if len(meal.options) != 5:
                return (
                    f"Day {day.day_number} {meal.meal} has {len(meal.options)} "
                    f"options, needs EXACTLY 5 different real nearby places."
                )
            names_seen = {opt.name.strip().lower() for opt in meal.options}
            if len(names_seen) != len(meal.options):
                return (
                    f"Day {day.day_number} {meal.meal} has duplicate/repeated "
                    f"option names -- all 5 must be different real places."
                )
    return None


async def run_generate(trip_state: dict, hotels: list, weather: dict) -> Itinerary:
    start, end = trip_state.get("start_date"), trip_state.get("end_date")
    is_day_trip = bool(start) and start == end

    hotel_names = [h["displayName"]["text"] for h in hotels]
    budget = trip_state.get("budget")
    previous_budget = trip_state.get("previous_budget")

    budget_line = (
        f"Budget: ₹{budget} (Indian Rupees). The estimated_total_cost MUST NOT "
        f"exceed this budget unless it is genuinely impossible to do so realistically "
        f"(basic {'food + local transport' if is_day_trip else 'stay + food + local transport'}) "
        f"-- in that case get as close as possible and clearly say so in notes. "
        f"Do not casually go over budget."
        if budget
        else "Budget: not specified — suggest a reasonable range in Indian Rupees (₹) "
        "and mention it in notes. Do not claim a specific budget was given."
    )
    if previous_budget and budget and previous_budget > budget:
        budget_line += (
            f"\nThis budget was REDUCED from a previous ₹{previous_budget} to ₹{budget}. "
            f"Populate budget_cuts explaining exactly what was downgraded/removed to fit "
            f"the new lower budget."
        )

    weather = weather or {}
    is_forecast = weather.get("is_forecast", True)
    weather_note = (
        "This is a real forecast for the exact trip dates."
        if is_forecast
        else "This is historical data from last year (typical-conditions estimate); "
        "the year won't match the trip dates, which is expected."
    )

    previous_issues = trip_state.get("previous_issues_to_fix")
    issues_line = f"\nFix these specific issues from the last attempt: {previous_issues}" if previous_issues else ""

    if is_day_trip:
        hotel_line = (
            f"This is a ONE-DAY trip on {start}. No hotel is needed. "
            f"Set selected_hotel to \"{DAY_TRIP_HOTEL}\" and produce exactly 1 day."
        )
    else:
        hotel_line = f"Available hotels: {hotel_names}"

    user_content = f"""
Trip details: {trip_state}
User instructions you MUST follow: {trip_state.get('notes') or 'none'}
{budget_line}
All costs must be in Indian Rupees (₹).

{hotel_line}

Weather data ({weather_note}): {weather.get('daily', weather.get('error', {}))}
{issues_line}

Generate a complete day-wise itinerary with timed activities, travel time
between them, nearby meal recommendations (no cost), and a budget breakdown.
"""
    result = await call_structured(SYSTEM_PROMPT, user_content, Itinerary, validate_fn=_validate_itinerary_meals)

    if is_day_trip:
        result.selected_hotel = DAY_TRIP_HOTEL  # force, in case the LLM ignores it
        result.hotel_options = None
    else:
        # Deterministic, never LLM-invented: the real list that was actually
        # fetched, each flagged as selected or not against what the LLM chose.
        result.hotel_options = [
            HotelOption(
                name=h["displayName"]["text"],
                address=h.get("formattedAddress"),
                selected=(h["displayName"]["text"] == result.selected_hotel),
            )
            for h in hotels
        ]

    # Never trust the LLM for these -- always carry through exactly what the
    # user stated (or null), regardless of what it put in its own response.
    flight = trip_state.get("flight")
    result.flight = FlightDetails(**flight) if flight else None
    car_rental = trip_state.get("car_rental")
    result.car_rental = CarRentalDetails(**car_rental) if car_rental else None

    return result
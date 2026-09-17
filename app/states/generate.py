from typing import Optional
from pydantic import BaseModel, Field
from app.llm import call_structured

SYSTEM_PROMPT = """You are the itinerary generation step of a travel planning agent.
Given the trip details, weather forecast, and available hotels, create a
realistic day-by-day itinerary.

Rules:
- Only use hotels from the provided list, do not invent new ones.
- Keep activities realistic for the number of days and destination.
- Consider the weather: suggest indoor activities on days with high rain probability.
- All costs, budgets, and estimated_total_cost must be in Indian Rupees (₹).
- If a budget is stated, stay within it as a rough guide. If no budget is
  stated, suggest a reasonable, moderate-cost plan in Indian Rupees and
  mention the estimated total cost clearly in notes.
- Each day should have a short list of activities/places, not a wall of text.
"""


class DayPlan(BaseModel):
    day_number: int
    date: str
    summary: str = Field(description="One-line summary of the day's theme")
    activities: list[str] = Field(description="3-5 short activity/place bullet points")


class Itinerary(BaseModel):
    destination: str
    selected_hotel: str = Field(description="Name of the chosen hotel from the provided list")
    days: list[DayPlan]
    estimated_total_cost: float
    notes: Optional[str] = Field(default=None, description="Any caveats, e.g. weather warnings")


async def run_generate(trip_state: dict, hotels: list, weather: dict) -> Itinerary:
    hotel_names = [h["displayName"]["text"] for h in hotels]
    budget = trip_state.get("budget")
    budget_line = (
        f"Budget: ₹{budget} (Indian Rupees)"
        if budget
        else "Budget: not specified — suggest a reasonable range in Indian Rupees (₹) and mention it in notes."
    )

    user_content = f"""
Trip details: {trip_state}
{budget_line}
All costs must be in Indian Rupees (₹).

Available hotels: {hotel_names}

Weather forecast (daily max/min temp in Celsius, rain probability %):
{weather.get('daily', {})}

Generate a complete day-wise itinerary.
"""
    result = await call_structured(SYSTEM_PROMPT, user_content, Itinerary)
    return result
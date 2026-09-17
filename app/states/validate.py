from typing import TypedDict
from app.llm import call_structured
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are the validation step of a travel planning agent.
Given the trip details, the generated itinerary, and the weather forecast,
check the itinerary for problems.

Check for:
- Activities that don't fit the destination or are unrealistic.
- Estimated total cost significantly exceeding the stated budget.
- Outdoor activities scheduled on days with high rain probability without a warning.
- Hotel name not matching any of the hotels actually offered.
- Missing or incomplete day plans (wrong number of days vs trip duration).

If everything looks fine, return is_valid=True with an empty issues list.
Otherwise return is_valid=False and a list of specific, actionable issues
so the itinerary can be regenerated to fix them.
"""


class ValidationResult(BaseModel):
    is_valid: bool
    issues: list[str] = Field(default_factory=list)


async def run_validate(trip_state: dict, itinerary: dict, weather: dict) -> dict:
    user_content = f"""
Trip details: {trip_state}

Generated itinerary: {itinerary}

Weather forecast: {weather.get('daily', {})}

Validate this itinerary.
"""
    result = await call_structured(SYSTEM_PROMPT, user_content, ValidationResult)
    return {"is_valid": result.is_valid, "issues": result.issues}
from tavily import TavilyClient
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import call_structured

DEFAULT_MIN_COST_PER_PERSON_PER_DAY = 800


class BudgetEstimate(BaseModel):
    min_cost_per_person_per_day: int = Field(
        description="Realistic minimum budget travel cost per person per day, in INR, "
        "covering basic stay + food + local transport. A conservative/frugal estimate."
    )


_EXTRACT_SYSTEM_PROMPT = """You are extracting a realistic minimum daily travel
budget from search results.

Given search results about travel costs for a destination, estimate the
minimum realistic cost per person per day in INR (Indian Rupees) for a
frugal/budget traveler -- covering basic accommodation, food, and local
transport.

If the search results mention costs in other currencies, convert to INR
using a reasonable approximate exchange rate. If the results are vague or
don't give a clear number, make a reasonable conservative estimate based on
whatever context is available (e.g. general cost-of-living hints for that
country/region). Always return some number -- never leave this unanswered.
"""


async def get_min_budget_per_day(destination: str) -> int:
    """
    Searches the web for a realistic minimum daily travel budget for the
    given destination, and returns an approximate INR figure per person per
    day. Falls back to a conservative default if search or parsing fails
    for any reason -- this must never raise, since it's a soft signal used
    to sanity-check user-provided budgets.
    """
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        search_result = client.search(
            query=f"average minimum daily budget travel cost per person {destination} India rupees",
            max_results=5,
        )

        snippets = "\n\n".join(
            f"{r.get('title', '')}: {r.get('content', '')}"
            for r in search_result.get("results", [])
        )

        if not snippets.strip():
            return DEFAULT_MIN_COST_PER_PERSON_PER_DAY

        result = await call_structured(
            _EXTRACT_SYSTEM_PROMPT,
            f"Destination: {destination}\n\nSearch results:\n{snippets}",
            BudgetEstimate,
        )
        return result.min_cost_per_person_per_day

    except Exception:
        # Search/parsing failure should never block trip planning --
        # fall back to the conservative default.
        return DEFAULT_MIN_COST_PER_PERSON_PER_DAY
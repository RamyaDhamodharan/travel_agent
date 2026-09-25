from tavily import TavilyClient
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import call_structured


class PlaceItem(BaseModel):
    name: str = Field(description="Name of the specific place/attraction.")
    description: str = Field(
        description="One or two sentences describing what it is and why it's "
        "worth visiting -- distilled from the search results, in your own words."
    )
    source_url: str | None = Field(
        default=None,
        description="The URL of the search result this place's information came "
        "from, if one is available in the provided results. Omit if unknown.",
    )


class PlaceCategory(BaseModel):
    category: str = Field(
        description="A short category label grouping similar places together, "
        "e.g. 'Religious and Historical Sites', 'Natural Attractions and "
        "Waterfalls', 'Wildlife and Nature', 'Museums and Cultural Spots', "
        "'Unique Experiences'. Choose labels that fit what's actually in the "
        "search results -- don't force a fixed category list."
    )
    places: list[PlaceItem] = Field(min_length=1, max_length=8)


class PlacesGuide(BaseModel):
    destination: str
    intro: str = Field(
        description="A one-sentence overview of what kind of trip this "
        "destination offers (e.g. blend of temples, waterfalls, wildlife)."
    )
    categories: list[PlaceCategory] = Field(min_length=1, max_length=6)


_EXTRACT_SYSTEM_PROMPT = """You are a travel guide assistant. Given raw web
search results about places to visit in a destination, organize them into a
clean, categorized sightseeing guide.

Group similar places under sensible category headings (religious/historical
sites, waterfalls/nature, wildlife, museums/cultural spots, unique local
experiences, etc -- pick whatever categories genuinely fit the results).
Write each place's description in your own words, concise (1-2 sentences).
Attach the source_url for a place when the search results make it clear
which result it came from; otherwise omit it. Do not invent places that
aren't supported by the search results. Skip anything that's not an actual
point of interest (e.g. generic booking sites, unrelated news)."""


def _fallback_guide(destination: str) -> PlacesGuide:
    return PlacesGuide(
        destination=destination,
        intro=f"Couldn't fetch live results for {destination} right now -- try again in a moment.",
        categories=[],
    )


async def get_places_guide(destination: str) -> PlacesGuide:
    """
    Searches the web for top places to visit in the given destination and
    returns a categorized guide (religious sites, waterfalls, wildlife,
    museums, unique experiences, etc). Falls back to an empty guide if
    search or parsing fails -- this must never raise, since it backs an
    optional "explore" feature, not core trip generation.
    """
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        search_result = client.search(
            query=f"best places to visit in {destination} tourist attractions "
            f"temples waterfalls museums things to do",
            max_results=12,
            search_depth="advanced",
        )

        results = search_result.get("results", [])
        if not results:
            return _fallback_guide(destination)

        snippets = "\n\n".join(
            f"TITLE: {r.get('title', '')}\nURL: {r.get('url', '')}\n"
            f"CONTENT: {r.get('content', '')}"
            for r in results
        )

        guide = await call_structured(
            _EXTRACT_SYSTEM_PROMPT,
            f"Destination: {destination}\n\nSearch results:\n{snippets}",
            PlacesGuide,
        )
        # Make sure the destination name is exactly what was asked for,
        # regardless of how the model normalized it.
        guide.destination = destination
        return guide

    except Exception:
        return _fallback_guide(destination)

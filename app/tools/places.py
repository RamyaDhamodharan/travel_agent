import httpx
from app.config import settings


async def search_places(query: str, max_results: int = 5) -> list[dict]:
    """
    Calls Google Places API (Text Search, New) to find places matching
    a free-text query like "hotels in Goa" or "attractions in Goa".
    """
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": settings.google_maps_api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.priceLevel,places.location",
    }
    payload = {
        "textQuery": query,
        "maxResultCount": max_results,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("places", [])
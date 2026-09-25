import asyncio
import logging
import httpx

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
HEADERS = {"User-Agent": "TravelPlanningAgent/1.0 (educational project)"}


async def geocode_destination(destination: str) -> dict | None:
    """Free geocoding via OpenStreetMap Nominatim. No API key."""
    params = {"q": destination, "format": "json", "limit": 1}
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(NOMINATIM_URL, params=params, headers=HEADERS)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                "Nominatim failed: status=%s body=%s",
                e.response.status_code, e.response.text[:300],
            )
            raise
        results = response.json()
        if not results:
            return None
        r = results[0]
        return {
            "latitude": float(r["lat"]),
            "longitude": float(r["lon"]),
            "display_name": r["display_name"],
        }


async def _query_mirror(client: httpx.AsyncClient, url: str, query: str) -> dict:
    response = await client.post(url, data={"data": query}, headers=HEADERS)
    response.raise_for_status()
    return response.json()


OVERPASS_TIMEOUT = 25  # small-town/around queries can be slow on public mirrors


async def _overpass(query: str) -> dict:
    """Race all mirrors; return the first successful response."""
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=OVERPASS_TIMEOUT) as client:
        tasks = [
            asyncio.create_task(_query_mirror(client, url, query))
            for url in OVERPASS_URLS
        ]
        try:
            for fut in asyncio.as_completed(tasks):
                try:
                    return await fut
                except Exception as e:  # this mirror failed, wait for the next
                    logger.error("Overpass mirror failed: %r", e)
                    last_error = e
        finally:
            for t in tasks:
                t.cancel()
    raise last_error or RuntimeError("Overpass API unavailable")


async def find_hotels_nearby(
    latitude: float, longitude: float, radius_meters: int = 8000, max_results: int = 5
) -> list[dict]:
    query = f"""
    [out:json][timeout:20];
    (
      node["tourism"="hotel"](around:{radius_meters},{latitude},{longitude});
      way["tourism"="hotel"](around:{radius_meters},{latitude},{longitude});
    );
    out center {max_results};
    """

    data = await _overpass(query)

    hotels = []
    for el in data.get("elements", [])[:max_results]:
        tags = el.get("tags", {})
        name = tags.get("name", "Unnamed Hotel")
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        address_parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:city", ""),
        ]
        address = ", ".join(p for p in address_parts if p) or "Address not available"
        hotels.append({
            "displayName": {"text": name},
            "location": {"latitude": lat, "longitude": lon},
            "formattedAddress": address,
        })
    return hotels
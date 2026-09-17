from app.tools.places import search_places
from app.tools.weather import get_weather_forecast


async def run_fetch_data(trip_state: dict) -> dict:
    """
    Given a complete trip_state (destination, dates, etc.), fetches:
    - Geocoded location for the destination (via Places search)
    - Weather forecast for the trip dates (using those coordinates)
    - A few hotel suggestions

    Returns a dict of fetched data to merge into the graph state.
    """
    destination = trip_state["destination"]

    location_results = await search_places(destination, max_results=1)
    if not location_results:
        return {"fetch_error": f"Could not find location data for {destination}"}

    location = location_results[0]["location"]
    latitude = location["latitude"]
    longitude = location["longitude"]

    hotels = await search_places(f"hotels in {destination}", max_results=3)

    weather = None
    try:
        weather = await get_weather_forecast(
            latitude=latitude,
            longitude=longitude,
            start_date=trip_state["start_date"],
            end_date=trip_state["end_date"],
        )
    except Exception as e:
        weather = {"error": f"Weather unavailable: {e}"}

    return {
        "hotels": hotels,
        "weather": weather,
        "destination_coordinates": {"latitude": latitude, "longitude": longitude},
    }
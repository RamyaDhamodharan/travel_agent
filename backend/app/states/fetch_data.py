import asyncio
import logging

from app.tools.places import geocode_destination, find_hotels_nearby
from app.tools.weather import get_weather_forecast

logger = logging.getLogger(__name__)

DAY_TRIP_HOTEL = "N/A (one-day trip, no hotel needed)"


def _is_day_trip(trip_state: dict) -> bool:
    start, end = trip_state.get("start_date"), trip_state.get("end_date")
    return bool(start) and start == end


async def _no_hotels() -> list:
    return []


async def run_fetch_data(trip_state: dict) -> dict:
    """
    Geocode first, then fetch hotels + weather in parallel.
    One-day trips (start_date == end_date) skip the hotel search entirely.
    Hotel/weather failures are NOT fatal. Only geocoding failure returns fetch_error.
    """
    destination = trip_state["destination"]
    day_trip = _is_day_trip(trip_state)

    try:
        location = await geocode_destination(destination)
    except Exception as e:
        logger.error("Geocode failed for %s: %r", destination, e)
        return {"fetch_error": f"Couldn't look up {destination} right now ({type(e).__name__}). Please try again in a moment."}

    if location is None:
        return {"fetch_error": f"Could not find location data for {destination}"}

    latitude = location["latitude"]
    longitude = location["longitude"]

    hotels_result, weather_result = await asyncio.gather(
        _no_hotels() if day_trip else find_hotels_nearby(latitude, longitude, max_results=3),
        get_weather_forecast(
            latitude=latitude,
            longitude=longitude,
            start_date=trip_state["start_date"],
            end_date=trip_state["end_date"],
        ),
        return_exceptions=True,
    )

    hotel_note = None
    if day_trip:
        # Placeholder keeps downstream code (cache check, generate) working.
        hotels = [{
            "displayName": {"text": DAY_TRIP_HOTEL},
            "location": {"latitude": latitude, "longitude": longitude},
        }]
    else:
        if isinstance(hotels_result, Exception):
            logger.error("Hotels failed for %s: %r", destination, hotels_result)
            hotels = []
            hotel_note = "Live hotel data was unavailable."
        else:
            hotels = hotels_result

        if not hotels:
            hotels = [{
                "displayName": {"text": f"Hotel in {destination} (no listings found)"},
                "location": {"latitude": latitude, "longitude": longitude},
            }]

    if isinstance(weather_result, Exception):
        logger.error("Weather failed for %s: %r", destination, weather_result)
        weather = {"error": f"Weather unavailable: {weather_result}"}
    else:
        weather = weather_result

    result = {
        "hotels": hotels,
        "weather": weather,
        "destination_coordinates": {"latitude": latitude, "longitude": longitude},
    }
    if hotel_note:
        result["hotel_note"] = hotel_note
    return result
import httpx
from datetime import date


async def get_weather_forecast(latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    """
    Calls Open-Meteo's free forecast API for the given coordinates and date
    range. No API key required, no rate limit for non-commercial use.

    The forecast API only predicts ~16 days ahead. For dates further out,
    this falls back to the free Historical/Archive API using the SAME
    calendar dates from last year, as an approximation of typical weather.
    The result is tagged with "is_forecast": False so callers/prompts can
    describe it as "typical" rather than an exact prediction.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)

        if response.status_code == 400:
            # Likely "date too far in the future" -- fall back to historical
            # data from the same dates last year as a typical-weather estimate.
            return await _get_historical_fallback(client, latitude, longitude, start_date, end_date)

        response.raise_for_status()
        data = response.json()
        data["is_forecast"] = True
        return data


async def _get_historical_fallback(
    client: httpx.AsyncClient, latitude: float, longitude: float, start_date: str, end_date: str
) -> dict:
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError:
        return {"error": "Invalid date format", "is_forecast": False}

    last_year_start = start.replace(year=start.year - 1)
    last_year_end = end.replace(year=end.year - 1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": last_year_start.isoformat(),
        "end_date": last_year_end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
    }

    response = await client.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    data["is_forecast"] = False
    data["note"] = (
        f"Trip dates are too far ahead for an exact forecast. "
        f"Showing typical weather based on the same dates last year "
        f"({last_year_start} to {last_year_end})."
    )
    return data
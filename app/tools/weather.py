import httpx


async def get_weather_forecast(latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    """
    Calls Open-Meteo's free forecast API for the given coordinates and date range.
    No API key required.
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
        response.raise_for_status()
        return response.json()
import datetime
import requests
from google.adk.tools import FunctionTool
from google.adk.agents import Agent



def geocode_city(name: str) -> dict:
    """
    Geocodes a city name to latitude, longitude, and timezone using Open-Meteo Geocoding API.

    Endpoint:
      GET https://geocoding-api.open-meteo.com/v1/search
      • name (string): search term, e.g. "Rockville MD"
      • count (int): number of results (1)
      • language (string): "en"
      • format (string): "json"

    Returns:
      { status: "success", result: { name, latitude, longitude, timezone } }
      or { status: "error", error_message }
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": name, "count": 1, "language": "en", "format": "json"}
    try:
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        items = res.json().get("results") or []
        if not items:
            return {"status": "error", "error_message": f"City '{name}' not found."}
        city = items[0]
        return {
            "status": "success",
            "result": {
                "name": city.get("name"),
                "latitude": city.get("latitude"),
                "longitude": city.get("longitude"),
                "timezone": city.get("timezone"),
            }
        }
    except requests.RequestException as e:
        return {"status": "error", "error_message": f"Geocoding error: {e}"}


def get_weather(city: str) -> dict:
    """
    Returns current weather for a city using Open-Meteo Forecast API.
    """
    geo = geocode_city(city)
    if geo.get("status") != "success":
        return {"status": "error", "error_message": geo.get("error_message")}

    lat = geo["result"]["latitude"]
    lon = geo["result"]["longitude"]
    name = geo["result"]["name"]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "relativehumidity_2m",
        "timezone": "auto"
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        cw = data.get("current_weather", {})
        # Align humidity
        times = data.get("hourly", {}).get("time", [])
        hums = data.get("hourly", {}).get("relativehumidity_2m", [])
        humidity = None
        tnow = cw.get("time")
        if tnow in times:
            humidity = hums[times.index(tnow)]
        else:
            humidity = hums[0] if hums else None

        weather_map = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 95: "Thunderstorm"
        }
        desc = weather_map.get(cw.get("weathercode"), "Unknown")
        tc = cw.get("temperature")
        tf = (tc * 9/5 + 32) if isinstance(tc, (int, float)) else None

        report = (
            f"The weather in {name} is {desc} with {tc:.1f}°C"
            f" ({tf:.1f}°F), humidity {humidity}%"
            f" and wind {cw.get('windspeed', 'N/A')} km/h."
        )
        return {"status": "success", "report": report}
    except requests.RequestException as e:
        return {"status": "error", "error_message": f"Weather lookup failed: {e}"}


def get_current_time(city: str) -> dict:
    """
    Returns current local time for a city using TimeAPI.io.

    Endpoint:
      GET https://www.timeapi.io/api/Time/current/zone
      • timeZone (string): IANA timezone identifier, e.g. "America/New_York"

    Steps:
      1) Call geocode_city to get timezone
      2) Query TimeAPI.io for current time in that zone

    Returns:
      { status: "success", report }
      or { status: "error", error_message }
    """
    geo = geocode_city(city)
    if geo.get("status") != "success":
        return {"status": "error", "error_message": geo.get("error_message")}
    tz = geo["result"]["timezone"]
    url = "https://www.timeapi.io/api/Time/current/zone"
    params = {"timeZone": tz}
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        dt_str = data.get("dateTime") or data.get("date") + "T" + data.get("time")
        dt = datetime.datetime.fromisoformat(dt_str)
        report = dt.strftime(f"%Y-%m-%d %H:%M:%S ({tz})")
        return {"status": "success", "report": report}
    except requests.RequestException as e:
        return {"status": "error", "error_message": f"Time lookup failed: {e}"}
    except Exception as e:
        return {"status": "error", "error_message": f"Time parsing failed: {e}"}


# Agent definition
root_agent = Agent(
    name="weather_time_agent",
    model="gemini-1.5-flash",
    description="A helpful assistant that provides real-time weather and local time for any city.",
    instruction=(
        '''You are a friendly and helpful agent who can answer user questions about the weather and time in cities around the world.

        - When a user asks for the weather, call the `get_weather(city)` tool with the specified city.
        - When a user asks for the time, call the `get_current_time(city)` tool with the specified city.
        - If a tool call is successful, use the `report` from the output to give a clear and friendly answer.
        - If a tool call fails (e.g., city not found), use the `error_message` to inform the user politely.
        - Always use the information from the tools to answer the user's question.'''
    ),
    tools=[get_weather, get_current_time]
)

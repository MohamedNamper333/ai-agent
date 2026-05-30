"""Example plugin - Weather tool"""

from plugins import Plugin


class WeatherPlugin(Plugin):
    name = "weather"
    description = "Get weather information for a city"

    def get_tools(self):
        return [
            {
                "name": "get_weather",
                "description": "Get current weather for a city. Params: city",
                "func": self.get_weather,
            }
        ]

    def get_weather(self, city: str) -> str:
        try:
            import requests
            url = f"https://wttr.in/{city}?format=%C+%t+%w+%h"
            r = requests.get(url, timeout=10)
            return f"Weather in {city}: {r.text.strip()}"
        except Exception as e:
            return f"Error fetching weather: {e}"

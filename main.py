import os
import requests
import json
import re
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="City Activity Advisor")

class StandardResponse(BaseModel):
    status: str
    data: dict | list | str | None = None
    message: str | None = None

class RecommendationRequest(BaseModel):
    city: str
    interest: str

def get_coordinates(city: str) -> tuple[float, float, str] | None:
    """
    Get latitude, longitude, and full name for a city using Open-Meteo Geocoding API.
    """
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json().get("results")
        if not results:
            return None
        res = results[0]
        return res["latitude"], res["longitude"], f"{res['name']}, {res.get('country', '')}"
    except Exception:
        return None

def get_weather(lat: float, lon: float) -> dict | None:
    """
    Get current weather for coordinates using Open-Meteo API.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("current_weather")
    except Exception:
        return None

@app.get("/", response_model=StandardResponse)
async def root():
    """
    Root endpoint to verify the API is running.
    """
    try:
        return StandardResponse(
            status="ok",
            data={"message": "Welcome to the City Activity Advisor API"}
        )
    except Exception as e:
        return StandardResponse(status="error", message=str(e))

@app.post("/recommend", response_model=StandardResponse)
async def recommend(request: RecommendationRequest):
    """
    Generate activity recommendations based on city, interest, and real-time weather.
    """
    try:
        # 1. Get Coordinates
        coord_data = get_coordinates(request.city)
        if not coord_data:
            return StandardResponse(status="error", message=f"Could not find coordinates for city: {request.city}")
        
        lat, lon, full_city_name = coord_data

        # 2. Get Weather
        weather = get_weather(lat, lon)
        if not weather:
            return StandardResponse(status="error", message=f"Could not fetch weather for {full_city_name}")

        # 3. Ask Gemini
        if not GEMINI_API_KEY:
            return StandardResponse(status="error", message="Gemini API key is not configured in environment variables.")

        prompt = f"""
        You are a city activity advisor.
        Location: {full_city_name}
        Current Weather: Temperature {weather['temperature']}°C, Windspeed {weather['windspeed']} km/h, Weathercode {weather['weathercode']}
        User's interest: {request.interest}
        
        Provide exactly 3 specific real-world activity recommendations for this city given the current weather. 
        Format your response ONLY as a JSON object with a 'recommendations' list. 
        Each recommendation must have 'name', 'description', and 'reason'.
        No other text outside the JSON.
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        # Parse JSON from response
        try:
            # Extract JSON from potential code blocks
            text = response.text
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                recommendations_data = json.loads(match.group(1))
            else:
                recommendations_data = json.loads(text)
        except Exception:
            recommendations_data = {"raw_text": response.text}
        
        return StandardResponse(
            status="ok",
            data={
                "city": full_city_name,
                "weather": weather,
                "recommendations": recommendations_data.get("recommendations", recommendations_data)
            }
        )
    except Exception as e:
        return StandardResponse(status="error", message=str(e))

@app.get("/health", response_model=StandardResponse)
async def health_check():
    """
    Check the health of the application and its dependencies.
    """
    try:
        return StandardResponse(status="ok", data={"health": "stable"})
    except Exception as e:
        return StandardResponse(status="error", message=str(e))

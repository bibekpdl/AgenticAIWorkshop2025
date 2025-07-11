"""Food Assistant Streamlit App

A simple Streamlit UI that wraps the Food Assistant SequentialAgent pipeline.

"""

import asyncio
import logging
import sqlite3
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
import streamlit as st

# ─── ENV & LOGGING ────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ─── CONSTANTS ────────────────────────────────────────────────────────────
GEMINI = "gemini-2.0-flash"
APP_NAME = "food_assistant_app"
USER_ID = "streamlit_user"
SESSION_ID = "streamlit_session"

# ─── TOOL FUNCTIONS ───────────────────────────────────────────────────────

def get_recipe_details(dish_name: str) -> dict:
    """Search for a recipe in the local SQLite database based on the dish name."""
    try:
        conn = sqlite3.connect("../13k-recipes.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT Title, Ingredients, Instructions FROM recipes WHERE Title LIKE ?",
            (f"%{dish_name}%",),
        )
        recipe = cursor.fetchone()
        conn.close()
        if recipe:
            return {
                "title": recipe[0],
                "ingredients": recipe[1],
                "instructions": recipe[2],
            }
        return {"error": "Recipe not found in the local database."}
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}


def get_nutrition_data(query: str) -> dict:
    """Fallback method to get nutritional information from Open Food Facts API."""
    try:
        search_url = (
            "https://world.openfoodfacts.org/cgi/search.pl?search_terms="
            f"{query}&search_simple=1&json=1"
        )
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("products") and len(data["products"]) > 0:
                product = data["products"][0]
                return {
                    "product_name": product.get("product_name", "N/A"),
                    "nutriments": product.get("nutriments", {}),
                    "categories": product.get("categories", "N/A"),
                    "ingredients_text": product.get("ingredients_text", "N/A"),
                }
        return {"error": "Could not retrieve nutritional data from the API."}
    except Exception as e:
        return {"error": f"API error: {str(e)}"}

# ─── AGENTS ───────────────────────────────────────────────────────────────

recipe_finder_agent = LlmAgent(
    name="RecipeFinderAgent",
    model=GEMINI,
    instruction="""You are a recipe finder agent. Your task is to get a recipe for the user.
    1. First, use the `get_recipe_details` tool to find the recipe in the database.
    2. If the tool returns an error or no recipe is found, create a reasonable recipe based on your own knowledge for the requested dish.
    3. Output the complete recipe, including title, a list of ingredients, and step-by-step instructions.""",
    description="Finds or creates a recipe based on user's request.",
    tools=[get_recipe_details],
    output_key="recipe_result",
)

nutrition_analyst_agent = LlmAgent(
    name="NutritionAnalystAgent",
    model=GEMINI,
    instruction="""You are a nutrition analyst agent.
    **Recipe to Analyze:**
    {recipe_result}

    **Task:**
    1. From the ingredients list above, identify the main components.
    2. For each major ingredient, use the `get_nutrition_data` tool to find its nutritional information.
    3. Synthesize the collected data into a summary for the entire dish.
    4. Present key nutrition facts like calories, protein, carbohydrates, and fat.
    **Output:** Provide a concise nutritional summary. If you cannot analyze the recipe, state why.""",
    description="Analyzes the nutritional content of the recipe.",
    tools=[get_nutrition_data],
    output_key="nutrition_result",
)

# Load allergen instruction from file
with open("allergen_agent_instruction.txt", "r", encoding="utf-8") as file:
    allergen_instruction_text = file.read()

allergen_alternative_agent = LlmAgent(
    name="AllergenAlternativeAgent",
    model=GEMINI,
    instruction=allergen_instruction_text,
    description="Identify allergens and suggest alternatives.",
    output_key="allergen_result",
)

final_response_agent = LlmAgent(
    name="FinalResponseAgent",
    model=GEMINI,
    instruction="""You are the final response coordinator. Your task is to assemble a complete and well-formatted answer for the user using the information gathered by the previous agents.

    **Recipe Data:**
    {recipe_result}

    **Nutrition Analysis:**
    {nutrition_result}

    **Allergen Analysis:**
    {allergen_result}

    **Task:**
    Combine all the information above into a single, cohesive response. Use the following markdown format:

    ### Recipe: <Recipe Title>
    **Ingredients:**
    - ...
    **Instructions:**
    1. ...

    ---
    ### 🥗 Nutritional Information
    ...

    ---
    ### ⚠️ Allergen Information & Alternatives
    ...

    Ensure the final output is clear, organized, and easy to read.""",
    description="Formats the final response from all pipeline stages.",
    output_key="final_response",
)

food_assistant_pipeline = SequentialAgent(
    name="FoodAssistantPipeline",
    sub_agents=[
        recipe_finder_agent,
        nutrition_analyst_agent,
        allergen_alternative_agent,
        final_response_agent,
    ],
    description="Executes a sequence of recipe finding, nutritional analysis, and allergen checking.",
)

# The root agent required by ADK runners
root_agent = food_assistant_pipeline

# ─── SESSION & RUNNER HELPERS ────────────────────────────────────────────

async def setup_session_and_runner():
    """Create (or fetch) a session and return the Runner instance."""
    service = InMemorySessionService()
    await service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    return Runner(agent=root_agent, app_name=APP_NAME, session_service=service)


async def _call_agent_async(query: str) -> str:
    """Internal async helper that returns the agent's final response as markdown."""
    runner = await setup_session_and_runner()
    user_msg = types.Content(role="user", parts=[types.Part(text=query)])
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg)

    final_text = ""
    async for ev in events:
        if ev.is_final_response():
            final_text = ev.content.parts[0].text
    return final_text


def call_agent(query: str) -> str:
    """Synchronous wrapper for Streamlit that executes the async pipeline."""
    return asyncio.run(_call_agent_async(query))

# ─── STREAMLIT UI ────────────────────────────────────────────────────────

st.set_page_config(page_title="Food Assistant", page_icon="🍲")
st.title("🍲 Food Assistant")
st.markdown(
    "Ask for a recipe, nutrition facts, or allergen information. Example: *Give me a gluten-free pancake recipe with nutrition facts and allergen info.*"
)

query = st.text_input("Your request:")

if st.button("Get Response") and query:
    with st.spinner("Cooking up the best answer..."):
        try:
            response_md = call_agent(query)
            st.markdown(response_md, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

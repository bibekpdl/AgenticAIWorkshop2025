# Food Assistant Agent - Multi-agent system for recipes, nutrition, and allergen analysis

import sqlite3
import requests
import json
from google.adk.agents import LlmAgent
from google import genai
import logging

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# --- Constants ---
GEMINI = "gemini-2.0-flash"


# --- Tool Functions ---


def get_recipe_details(dish_name: str) -> dict:
    """Search for a recipe in the local SQLite database based on the dish name."""
    try:
        conn = sqlite3.connect("13k-recipes.db")
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
        # Try searching for the product first
        search_url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={query}&search_simple=1&json=1"
        response = requests.get(search_url)
       
        if response.status_code == 200:
            data = response.json()
            if data.get("products") and len(data["products"]) > 0:
                product = data["products"][0]
                return {
                    "product_name": product.get("product_name", "N/A"),
                    "nutriments": product.get("nutriments", {}),
                    "categories": product.get("categories", "N/A"),
                    "ingredients_text": product.get("ingredients_text", "N/A")
                }
       
        # If search fails, try direct product lookup
        url = f"https://world.openfoodfacts.org/api/v2/product/{query}.json"
        response = requests.get(url)
        if response.status_code == 200 and response.json().get("status") == 1:
            product = response.json()["product"]
            return {
                "product_name": product.get("product_name", "N/A"),
                "nutriments": product.get("nutriments", {}),
                "categories": product.get("categories", "N/A"),
                "ingredients_text": product.get("ingredients_text", "N/A")
            }
       
        return {"error": "Could not retrieve nutritional data from the API."}
    except Exception as e:
        return {"error": f"API error: {str(e)}"}


# --- Sub Agent 1: Recipe Finder ---
recipe_finder_agent = LlmAgent(
    name="RecipeFinder",
    model=GEMINI,
    instruction="""
        You are a recipe finder agent. Your task is to help users find recipes.
       
        For recipe searches:
        1. First, try to find the recipe in the local database by using get_recipe_details
        2. If not found, create a reasonable recipe yourself based on your knowledge
        3. Always provide: title, ingredients list, and cooking instructions
        4. Format the response clearly with sections for ingredients and instructions
       
        When presenting recipes, make sure to:
        - List all ingredients with measurements
        - Provide step-by-step cooking instructions
        - Include cooking time and serving size if available
        - Mention any important cooking tips or variations
       
    """,
    description="Finds recipes using the internal DB or creates them based on knowledge.",
    tools=[get_recipe_details],
    output_key="recipe_result",
)


# --- Sub Agent 2: Nutrition Analyst ---


nutrition_analyst_agent = LlmAgent(
    name="NutritionAnalyst",
    model=GEMINI,
    instruction="""
        You are a nutrition analyst agent. Your task is to provide nutritional information about food items.
        
        For nutrition analysis:
        1. Get the food item or ingredients from state['recipe_result'] or user input
        2. Use the Open Food Facts API to get nutrition data
        3. Present key nutrition facts including calories, macronutrients, vitamins, and minerals
        
        When presenting nutrition information:
        - Focus on calories, protein, carbs, fat, fiber, and sodium
        - Include important vitamins and minerals when available
        - Mention any notable health benefits or concerns
        - Provide context for daily values when possible
        - Be clear about serving sizes
        
    """,
    description="Finds nutritional facts from Open Food Facts API.",
    tools=[get_nutrition_data],
    output_key="nutrition_result",
)



# Load instruction from external file
with open("myRecipeAgent2/allergen_agent_instruction.txt", "r", encoding="utf-8") as file:
    instruction_text = file.read()

# Define the agent with dynamic instruction
allergen_alternative_agent = LlmAgent(
    name="AllergenAlternative",
    model=GEMINI,
    instruction=instruction_text,
    description="Identify allergens and suggest alternatives.",
    output_key="allergen_result",
)

# --- LLM Agent Workflow ---
food_assistant_coordinator_agent = LlmAgent(
    name="food_assistant_coordinator_agent",
    model=GEMINI,
    instruction="""
        You are the main food assistant coordinator. Your role is to understand user requests and coordinate with specialized sub-agents.
       
        Determine the user's goal and route to appropriate sub-agents:
        - Use RecipeFinder for recipe searches and cooking instructions
        - Use NutritionAnalyst for nutritional information and health data
        - Use AllergenAlternative for allergen identification and safe substitutions
       
        After sub-agents complete their tasks, you can access their results:
        - state['recipe_result'] - Recipe information from RecipeFinder
        - state['nutrition_result'] - Nutrition data from NutritionAnalyst  
        - state['allergen_result'] - Allergen analysis from AllergenAlternative
       
        Your final response should:
        1. Combine relevant information from all sub-agents
        2. Format everything in clear, organized sections
        3. Provide comprehensive answers that address the user's needs
        4. Include practical tips and recommendations
        5. Always prioritize food safety, especially for allergen concerns
       
        Format your final response with clear sections:
        - **Recipe** (if applicable)
        - **Nutrition Information** (if applicable)
        - **Allergen Information & Alternatives** (if applicable)
        - **Additional Tips & Recommendations**
       
        Be helpful, informative, and always prioritize user safety.
    """,
    description="Routes requests to recipe, nutrition, or allergen agents and formats the final response.",
    sub_agents=[recipe_finder_agent, nutrition_analyst_agent, allergen_alternative_agent],
)


# --- Root Agent for the Runner ---
root_agent = food_assistant_coordinator_agent
# Food Assistant Agent - Converted to a Sequential Pipeline
import sqlite3
import requests
import json
from google.adk.agents import LlmAgent, SequentialAgent
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
        return {"error": "Could not retrieve nutritional data from the API."}
    except Exception as e:
        return {"error": f"API error: {str(e)}"}

# --- 1. Define Sub-Agents for Each Pipeline Stage ---

# Stage 1: Recipe Finder Agent
# Takes the initial user query to find or create a recipe.
recipe_finder_agent = LlmAgent(
    name="RecipeFinderAgent",
    model=GEMINI,
    instruction="""You are a recipe finder agent. Your task is to get a recipe for the user.
    1. First, use the `get_recipe_details` tool to find the recipe in the database.
    2. If the tool returns an error or no recipe is found, create a reasonable recipe based on your own knowledge for the requested dish.
    3. Output the complete recipe, including title, a list of ingredients, and step-by-step instructions.""",
    description="Finds or creates a recipe based on user's request.",
    tools=[get_recipe_details],
    output_key="recipe_result" # Stores output in state['recipe_result']
)

# Stage 2: Nutrition Analyst Agent
# Takes the recipe from the previous agent and analyzes its nutritional content.
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
    output_key="nutrition_result", # Stores output in state['nutrition_result']
)

# Stage 3: Allergen and Alternative Agent
# Takes the recipe and identifies potential allergens, suggesting alternatives.
# Load instruction from external file
with open("myRecipeAgent3/allergen_agent_instruction.txt", "r", encoding="utf-8") as file:
    instruction_text = file.read()


allergen_alternative_agent = LlmAgent(
    name="AllergenAlternative",
    model=GEMINI,
    instruction=instruction_text,
    description="Identify allergens and suggest alternatives.",
    output_key="allergen_result", # Stores output in state['allergen_result']

)

# allergen_alternative_agent = LlmAgent(
#     name="AllergenAlternativeAgent",
#     model=GEMINI,
#     instruction="""You are an allergen analysis agent.
#     **Recipe to Analyze:**
#     {recipe_result}

#     **Task:**
#     1. Carefully review the ingredients list from the recipe provided above.
#     2. Identify common allergens (e.g., nuts, dairy, gluten, soy, shellfish).
#     3. For each allergen found, suggest 1-2 safe and practical alternatives that would work well in the recipe.
#     4. If no common allergens are found, state "No common allergens identified."
#     **Output:** Present your findings as a clear, bulleted list.
#     """,
#     description="Identifies allergens in the recipe and suggests alternatives.",
#     output_key="allergen_result", # Stores output in state['allergen_result']
# )

# Stage 4: Final Response Formatter Agent
# Takes all the collected data from the state and formats a final, user-friendly response.
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

    Ensure the final output is clear, organized, and easy to read.
    """,
    description="Formats the final response from all pipeline stages.",
    output_key="final_response" # The final, formatted output
)


# --- 2. Create the SequentialAgent ---
# This agent orchestrates the pipeline by running the sub-agents in a fixed order.
food_assistant_pipeline = SequentialAgent(
    name="FoodAssistantPipeline",
    sub_agents=[
        recipe_finder_agent,
        nutrition_analyst_agent,
        allergen_alternative_agent,
        final_response_agent,
    ],
    description="Executes a sequence of recipe finding, nutritional analysis, and allergen checking.",
    # The agents will run in the order provided:
    # Finder -> Analyst -> Allergen -> Formatter
)

# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = food_assistant_pipeline
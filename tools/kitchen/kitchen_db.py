from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


# ======================
# Data Models
# ======================
@dataclass
class NutritionInfo:
    """Nutrition facts for an ingredient."""
    basis: str
    serving_size_g: Optional[float] = None
    calories_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    carbs_g: Optional[float] = None
    sugar_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sodium_mg: Optional[float] = None


@dataclass
class Ingredient:
    """Represents an ingredient in the kitchen."""
    name: str
    quantity: float  # in units of 100g
    expiry_date: Optional[str]
    storage_location: Optional[str]
    category: str
    nutrition: NutritionInfo


@dataclass
class RecipeIngredient:
    """Ingredient required for a recipe."""
    ingredient_name: str
    quantity: float  # in units of 100g


@dataclass
class CookingStep:
    """A single step in a recipe."""
    step_number: int
    description: str
    ingredients: List[str]
    action: str


@dataclass
class Recipe:
    """Represents a recipe."""
    name: str
    ingredients: List[RecipeIngredient]
    steps: List[CookingStep]
    allergens: List[str]
    taste: List[str]
    nutritional_characteristics: List[str]
    nutrition: Dict[str, float]


@dataclass
class ShoppingItem:
    """An item in a shopping list."""
    ingredient_name: str
    quantity: float  # in units of 100g


# ======================
# Kitchen Database Class
# ======================
class KitchenDB:
    """
    Kitchen Management System:
    - Manage ingredients inventory
    - Manage recipes catalog
    - Manage user menus and shopping lists
    """

    def __init__(self):
        # Ingredient catalog: ingredient_name -> Ingredient
        self.ingredients: Dict[str, Ingredient] = {}
        # Recipe catalog: recipe_name -> Recipe
        self.recipes: Dict[str, Recipe] = {}
        # User menus: user_id -> list of recipe names
        self.user_menus: Dict[str, List[str]] = {}
        # User shopping lists: user_id -> ingredient_name -> ShoppingItem
        self.user_shopping_lists: Dict[str, Dict[str, ShoppingItem]] = {}

    # ======================
    # Initialization Method
    # ======================
    def init_from_json(self, data: Dict[str, Any]) -> None:
        """
        Initialize the system from JSON data.

        Expected JSON structure:
        {
            "ingredients": [{...}, ...],
            "recipes": [{...}, ...],
            "user_menus": [{...}, ...],
            "user_shopping_lists": [{...}, ...]
        }
        """
        # Clear existing data
        self.ingredients.clear()
        self.recipes.clear()
        self.user_menus.clear()
        self.user_shopping_lists.clear()

        # 1. Initialize ingredients
        if "ingredients" in data:
            for ing_info in data["ingredients"]:
                ing_name = ing_info.get("name", "").lower()
                if not ing_name:
                    continue

                nutrition_info = ing_info.get("nutrition", {})
                nutrition_obj = NutritionInfo(
                    basis=nutrition_info.get("basis", "PER_100G"),
                    serving_size_g=nutrition_info.get("serving_size_g"),
                    calories_kcal=nutrition_info.get("calories_kcal"),
                    protein_g=nutrition_info.get("protein_g"),
                    fat_g=nutrition_info.get("fat_g"),
                    carbs_g=nutrition_info.get("carbs_g"),
                    sugar_g=nutrition_info.get("sugar_g"),
                    fiber_g=nutrition_info.get("fiber_g"),
                    sodium_mg=nutrition_info.get("sodium_mg")
                )

                ingredient = Ingredient(
                    name=ing_name,
                    quantity=ing_info.get("quantity", 0),
                    expiry_date=ing_info.get("expiry_date"),
                    storage_location=ing_info.get("storage_location"),
                    category=ing_info.get("category", "").lower(),
                    nutrition=nutrition_obj
                )
                self.ingredients[ing_name] = ingredient

        # 2. Initialize recipes
        if "recipes" in data:
            for recipe_info in data["recipes"]:
                recipe_name = recipe_info.get("name", "").lower()
                if not recipe_name:
                    continue

                # Parse ingredients
                ingredients_list = []
                for ing in recipe_info.get("ingredients", []):
                    ingredients_list.append(RecipeIngredient(
                        ingredient_name=ing.get("ingredient_name", "").lower(),
                        quantity=ing.get("quantity", 0)
                    ))

                # Parse steps
                steps_list = []
                for step in recipe_info.get("steps", []):
                    steps_list.append(CookingStep(
                        step_number=step.get("step_number", 0),
                        description=step.get("description", ""),
                        ingredients=step.get("ingredients", []),
                        action=step.get("action", "")
                    ))

                recipe = Recipe(
                    name=recipe_name,
                    ingredients=ingredients_list,
                    steps=steps_list,
                    allergens=[a.lower() for a in recipe_info.get("allergens", [])],
                    taste=[t.lower() for t in recipe_info.get("taste", [])],
                    nutritional_characteristics=[c.lower() for c in recipe_info.get("nutritional_characteristics", [])],
                    nutrition=recipe_info.get("nutrition", {})
                )
                self.recipes[recipe_name] = recipe

        # 3. Initialize user menus
        if "user_menus" in data:
            for menu_info in data["user_menus"]:
                user_id = menu_info.get("user_id")
                if not user_id:
                    continue
                self.user_menus[user_id] = [r.lower() for r in menu_info.get("recipes", [])]

        # 4. Initialize user shopping lists
        if "user_shopping_lists" in data:
            for shopping_info in data["user_shopping_lists"]:
                user_id = shopping_info.get("user_id")
                if not user_id:
                    continue

                self.user_shopping_lists[user_id] = {}
                for item in shopping_info.get("items", []):
                    ing_name = item.get("ingredient_name", "").lower()
                    self.user_shopping_lists[user_id][ing_name] = ShoppingItem(
                        ingredient_name=ing_name,
                        quantity=item.get("quantity", 0)
                    )

        print(f"✓ Initialization complete.")
        print(f"  - Ingredients: {len(self.ingredients)}")
        print(f"  - Recipes: {len(self.recipes)}")
        print(f"  - User menus: {len(self.user_menus)}")
        print(f"  - Shopping lists: {len(self.user_shopping_lists)}")

    # ======================
    # Menu Management Tools
    # ======================
    def add_recipe_to_menu(self, user_id: str, recipe_name: str) -> Dict[str, Any]:
        """Add a recipe to the user's daily menu."""
        recipe_key = recipe_name.lower()

        if recipe_key not in self.recipes:
            return {"status": "error", "message": f"Recipe '{recipe_name}' not found."}

        if user_id not in self.user_menus:
            self.user_menus[user_id] = []

        if recipe_key not in self.user_menus[user_id]:
            self.user_menus[user_id].append(recipe_key)
            return {"status": "success", "message": f"Recipe '{recipe_name}' added to menu.", "menu": self.user_menus[user_id]}
        else:
            return {"status": "success", "message": f"Recipe '{recipe_name}' already in menu.", "menu": self.user_menus[user_id]}

    def remove_recipe_from_menu(self, user_id: str, recipe_name: str) -> Dict[str, Any]:
        """Remove a recipe from the user's daily menu."""
        recipe_key = recipe_name.lower()

        if user_id not in self.user_menus:
            return {"status": "error", "message": f"No menu found for user '{user_id}'."}

        if recipe_key in self.user_menus[user_id]:
            self.user_menus[user_id].remove(recipe_key)
            return {"status": "success", "message": f"Recipe '{recipe_name}' removed from menu.", "menu": self.user_menus[user_id]}
        else:
            return {"status": "error", "message": f"Recipe '{recipe_name}' not found in menu.", "menu": self.user_menus[user_id]}

    def get_current_menu(self, user_id: str) -> Dict[str, Any]:
        """Retrieve the list of recipes planned for today."""
        if user_id not in self.user_menus:
            return {"user_id": user_id, "recipes": []}
        return {"user_id": user_id, "recipes": self.user_menus[user_id]}

    # ======================
    # Nutritional/Taste Aggregation Tools
    # ======================
    def tally_total_nutritional_characteristics(self, user_id: str, recipes: List[str]) -> Dict[str, Any]:
        """Aggregate nutritional characteristics for specified recipes."""
        all_characteristics = set()

        for recipe_name in recipes:
            recipe_key = recipe_name.lower()
            if recipe_key in self.recipes:
                all_characteristics.update(self.recipes[recipe_key].nutritional_characteristics)

        # Clean up empty strings
        all_characteristics = [c for c in all_characteristics if c]

        return {"user_id": user_id, "recipes": recipes, "nutritional_characteristics": sorted(list(all_characteristics))}

    def tally_total_tastes(self, user_id: str, recipes: List[str]) -> Dict[str, Any]:
        """Aggregate taste characteristics for specified recipes."""
        all_tastes = set()

        for recipe_name in recipes:
            recipe_key = recipe_name.lower()
            if recipe_key in self.recipes:
                all_tastes.update(self.recipes[recipe_key].taste)

        return {"user_id": user_id, "recipes": recipes, "tastes": sorted(list(all_tastes))}

    # ======================
    # Shopping List Management Tools
    # ======================
    def add_to_shopping_list(self, user_id: str, ingredient_name: str, quantity: float) -> Dict[str, Any]:
        """Add ingredients with quantities to the shopping list."""
        # Convert quantity to float if it's a string
        if isinstance(quantity, str):
            try:
                quantity = float(quantity)
            except ValueError:
                return {"status": "error", "message": "Quantity must be a valid number."}

        if quantity <= 0:
            return {"status": "error", "message": "Quantity must be greater than 0."}

        ing_key = ingredient_name.lower()

        if user_id not in self.user_shopping_lists:
            self.user_shopping_lists[user_id] = {}

        if ing_key in self.user_shopping_lists[user_id]:
            self.user_shopping_lists[user_id][ing_key].quantity += quantity
        else:
            self.user_shopping_lists[user_id][ing_key] = ShoppingItem(
                ingredient_name=ing_key,
                quantity=quantity
            )

        return {"status": "success", "message": f"Added {quantity} x {ingredient_name} to shopping list."}

    def remove_from_shopping_list(self, user_id: str, ingredient_name: str) -> Dict[str, Any]:
        """Remove a specific ingredient from the shopping list."""
        ing_key = ingredient_name.lower()

        if user_id not in self.user_shopping_lists:
            return {"status": "error", "message": f"No shopping list found for user '{user_id}'."}

        if ing_key in self.user_shopping_lists[user_id]:
            del self.user_shopping_lists[user_id][ing_key]
            return {"status": "success", "message": f"Removed '{ingredient_name}' from shopping list."}
        else:
            return {"status": "error", "message": f"'{ingredient_name}' not found in shopping list."}

    def get_current_shopping_list(self, user_id: str) -> Dict[str, Any]:
        """Retrieve the user's current shopping list."""
        if user_id not in self.user_shopping_lists:
            return {"user_id": user_id, "items": []}

        items = [
            {"ingredient_name": item.ingredient_name, "quantity": item.quantity}
            for item in self.user_shopping_lists[user_id].values()
        ]
        return {"user_id": user_id, "items": items}

    def compute_total_nutritions(self, user_id: str, ingredients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate total nutritional values for specified ingredients."""
        total_nutrition = {
            "calories_kcal": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "carbs_g": 0.0,
            "sugar_g": 0.0,
            "fiber_g": 0.0,
            "sodium_mg": 0.0
        }

        for item in ingredients:
            ing_name = item.get("ingredient_name", "").lower()
            quantity = item.get("quantity", 0)

            # Convert quantity to float if it's a string
            if isinstance(quantity, str):
                try:
                    quantity = float(quantity)
                except ValueError:
                    continue

            if ing_name in self.ingredients:
                ing = self.ingredients[ing_name]
                nut = ing.nutrition

                # Multiply by quantity (each unit is 100g)
                total_nutrition["calories_kcal"] += (nut.calories_kcal or 0) * quantity
                total_nutrition["protein_g"] += (nut.protein_g or 0) * quantity
                total_nutrition["fat_g"] += (nut.fat_g or 0) * quantity
                total_nutrition["carbs_g"] += (nut.carbs_g or 0) * quantity
                total_nutrition["sugar_g"] += (nut.sugar_g or 0) * quantity
                total_nutrition["fiber_g"] += (nut.fiber_g or 0) * quantity
                total_nutrition["sodium_mg"] += (nut.sodium_mg or 0) * quantity

        # Round values
        rounded_nutrition = {k: round(v, 2) for k, v in total_nutrition.items()}
        return {"user_id": user_id, "total_nutrition": rounded_nutrition}

    # ======================
    # Recipe Query Tools
    # ======================
    def get_cooking_steps(self, recipe_name: str) -> Dict[str, Any]:
        """Retrieve detailed step-by-step instructions for a recipe."""
        recipe_key = recipe_name.lower()

        if recipe_key not in self.recipes:
            return {"status": "error", "message": f"Recipe '{recipe_name}' not found."}

        recipe = self.recipes[recipe_key]
        steps = [
            {
                "step_number": step.step_number,
                "description": step.description,
                "ingredients": step.ingredients,
                "action": step.action
            }
            for step in recipe.steps
        ]
        return {"recipe_name": recipe_name, "steps": steps}

    def get_recipe_allergens(self, recipe_name: str) -> Dict[str, Any]:
        """Retrieve allergen information for a specific recipe."""
        recipe_key = recipe_name.lower()

        if recipe_key not in self.recipes:
            return {"status": "error", "message": f"Recipe '{recipe_name}' not found."}

        return {"recipe_name": recipe_name, "allergens": self.recipes[recipe_key].allergens}

    def find_recipes_by_allergen(self, allergen: str) -> Dict[str, Any]:
        """Find all recipes containing a specific allergen."""
        allergen_lower = allergen.lower()
        matching_recipes = [
            recipe.name for recipe in self.recipes.values()
            if allergen_lower in recipe.allergens
        ]
        return {"allergen": allergen, "recipes": matching_recipes}

    def get_recipe_taste(self, recipe_name: str) -> Dict[str, Any]:
        """Retrieve taste profile for a specific recipe."""
        recipe_key = recipe_name.lower()

        if recipe_key not in self.recipes:
            return {"status": "error", "message": f"Recipe '{recipe_name}' not found."}

        return {"recipe_name": recipe_name, "taste": self.recipes[recipe_key].taste}

    def find_recipes_by_taste(self, taste: str) -> Dict[str, Any]:
        """Find all recipes matching a specific taste profile."""
        taste_lower = taste.lower()
        matching_recipes = [
            recipe.name for recipe in self.recipes.values()
            if taste_lower in recipe.taste
        ]
        return {"taste": taste, "recipes": matching_recipes}

    def get_recipe_ingredients(self, recipe_name: str) -> Dict[str, Any]:
        """Retrieve the list of ingredients and quantities required for a recipe."""
        recipe_key = recipe_name.lower()

        if recipe_key not in self.recipes:
            return {"status": "error", "message": f"Recipe '{recipe_name}' not found."}

        recipe = self.recipes[recipe_key]
        ingredients = [
            {"ingredient_name": ing.ingredient_name, "quantity": ing.quantity}
            for ing in recipe.ingredients
        ]
        return {"recipe_name": recipe_name, "ingredients": ingredients}

    def find_recipes_by_ingredient(self, ingredient_name: str) -> Dict[str, Any]:
        """Find all recipes that require a specific ingredient."""
        ing_key = ingredient_name.lower()
        matching_recipes = []

        for recipe in self.recipes.values():
            for ing in recipe.ingredients:
                if ing.ingredient_name == ing_key:
                    matching_recipes.append(recipe.name)
                    break

        return {"ingredient_name": ingredient_name, "recipes": matching_recipes}

    def get_recipe_nutritional_characteristics(self, recipe_name: str) -> Dict[str, Any]:
        """Retrieve nutritional characteristic tags for a specific recipe."""
        recipe_key = recipe_name.lower()

        if recipe_key not in self.recipes:
            return {"status": "error", "message": f"Recipe '{recipe_name}' not found."}

        return {"recipe_name": recipe_name, "nutritional_characteristics": self.recipes[recipe_key].nutritional_characteristics}

    def find_recipes_by_nutritional_characteristics(self, characteristic: str) -> Dict[str, Any]:
        """Find all recipes matching specific nutritional characteristics."""
        char_lower = characteristic.lower()
        matching_recipes = [
            recipe.name for recipe in self.recipes.values()
            if char_lower in recipe.nutritional_characteristics
        ]
        return {"characteristic": characteristic, "recipes": matching_recipes}

    def get_all_recipe_names(self) -> Dict[str, Any]:
        """Retrieve the names of all available recipes."""
        return {"recipes": list(self.recipes.keys())}

    # ======================
    # Ingredient Query Tools
    # ======================
    def get_ingredient_shelf_life(self, ingredient_name: str) -> Dict[str, Any]:
        """Retrieve the expiration date for a specific ingredient."""
        ing_key = ingredient_name.lower()

        if ing_key not in self.ingredients:
            return {"status": "error", "message": f"Ingredient '{ingredient_name}' not found."}

        expiry = self.ingredients[ing_key].expiry_date
        return {"ingredient_name": ingredient_name, "expiry_date": expiry}

    def find_ingredients_by_expiry_date(self, date: str) -> Dict[str, Any]:
        """Find ingredients expiring after a specific date."""
        matching_ingredients = []

        for ing in self.ingredients.values():
            if ing.expiry_date and ing.expiry_date >= date:
                matching_ingredients.append(ing.name)

        return {"date": date, "ingredients": matching_ingredients}

    def get_ingredient_location(self, ingredient_name: str) -> Dict[str, Any]:
        """Retrieve the storage location for a specific ingredient."""
        ing_key = ingredient_name.lower()

        if ing_key not in self.ingredients:
            return {"status": "error", "message": f"Ingredient '{ingredient_name}' not found."}

        location = self.ingredients[ing_key].storage_location
        return {"ingredient_name": ingredient_name, "storage_location": location}

    def find_ingredients_by_location(self, location: str) -> Dict[str, Any]:
        """Find all ingredients stored in a specific location."""
        location_lower = location.lower()
        matching_ingredients = [
            ing.name for ing in self.ingredients.values()
            if ing.storage_location and ing.storage_location.lower() == location_lower
        ]
        return {"location": location, "ingredients": matching_ingredients}

    def get_ingredient_nutrition(self, ingredient_name: str) -> Dict[str, Any]:
        """Retrieve detailed nutritional facts for a specific ingredient."""
        ing_key = ingredient_name.lower()

        if ing_key not in self.ingredients:
            return {"status": "error", "message": f"Ingredient '{ingredient_name}' not found."}

        nut = self.ingredients[ing_key].nutrition
        nutrition_data = {
            "basis": nut.basis,
            "serving_size_g": nut.serving_size_g,
            "calories_kcal": nut.calories_kcal,
            "protein_g": nut.protein_g,
            "fat_g": nut.fat_g,
            "carbs_g": nut.carbs_g,
            "sugar_g": nut.sugar_g,
            "fiber_g": nut.fiber_g,
            "sodium_mg": nut.sodium_mg
        }
        return {"ingredient_name": ingredient_name, "nutrition": nutrition_data}

    def get_ingredient_quantity(self, ingredient_name: str) -> Dict[str, Any]:
        """Retrieve the current stock quantity of a specific ingredient."""
        ing_key = ingredient_name.lower()

        if ing_key not in self.ingredients:
            return {"status": "error", "message": f"Ingredient '{ingredient_name}' not found."}

        quantity = self.ingredients[ing_key].quantity
        return {"ingredient_name": ingredient_name, "quantity": quantity}

    def get_all_ingredient_names(self) -> Dict[str, Any]:
        """Retrieve the names of all available ingredients."""
        return {"ingredients": list(self.ingredients.keys())}

    def get_ingredients_by_category(self, category: str) -> Dict[str, Any]:
        """Retrieve ingredients belonging to a specific category."""
        category_lower = category.lower()
        matching_ingredients = [
            ing.name for ing in self.ingredients.values()
            if ing.category.lower() == category_lower
        ]
        return {"category": category, "ingredients": matching_ingredients}

    def find_ingredient_category(self, ingredient_name: str) -> Dict[str, Any]:
        """Find the category of a specific ingredient."""
        ing_key = ingredient_name.lower()

        if ing_key not in self.ingredients:
            return {"status": "error", "message": f"Ingredient '{ingredient_name}' not found."}

        category = self.ingredients[ing_key].category
        return {"ingredient_name": ingredient_name, "category": category}

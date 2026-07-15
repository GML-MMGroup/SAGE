#!/usr/bin/env python3
"""
Kitchen Database Interactive Test

Allows manual selection of initialization data and testing of all database operations.
"""

import sys
from typing import Dict, Any, List

from kitchen_db import KitchenDB
from kitchen_init import kitchen_init_data


# Map scenario numbers to their data
SCENARIOS = {
    1: ("Kitchen Main Data", kitchen_init_data),
}


def print_menu():
    """Print the main menu."""
    print("\n" + "=" * 60)
    print("KITCHEN DATABASE INTERACTIVE TEST")
    print("=" * 60)
    print("\n=== Database Management ===")
    print("[1] Initialize Database")
    print("[2] Show Catalog Summary")
    print("\n=== Ingredient Queries ===")
    print("[3] Get All Ingredients")
    print("[4] Get Ingredients by Category")
    print("[5] Find Ingredient Category")
    print("[6] Get Ingredient Nutrition")
    print("[7] Get Ingredient Quantity")
    print("[8] Get Ingredient Storage Location")
    print("[9] Find Ingredients by Location")
    print("[10] Get Ingredient Shelf Life")
    print("[11] Find Ingredients by Expiry Date")
    print("\n=== Recipe Queries ===")
    print("[12] Get All Recipes")
    print("[13] Search Recipe")
    print("[14] Get Recipe Ingredients")
    print("[15] Get Cooking Steps")
    print("[16] Get Recipe Allergens")
    print("[17] Find Recipes by Allergen")
    print("[18] Get Recipe Taste")
    print("[19] Find Recipes by Taste")
    print("[20] Get Recipe Nutritional Characteristics")
    print("[21] Find Recipes by Nutritional Characteristics")
    print("[22] Find Recipes by Ingredient")
    print("\n=== Menu Management ===")
    print("[23] Add Recipe to Menu")
    print("[24] Remove Recipe from Menu")
    print("[25] Get Current Menu")
    print("[26] Tally Nutritional Characteristics")
    print("[27] Tally Tastes")
    print("\n=== Shopping List Management ===")
    print("[28] Add to Shopping List")
    print("[29] Remove from Shopping List")
    print("[30] Get Shopping List")
    print("\n=== Calculations ===")
    print("[31] Compute Total Nutrition")
    print("\n[0] Exit")
    print("-" * 60)


def initialize_database() -> KitchenDB:
    """Let user select and initialize a scenario."""
    print("\nAvailable Scenarios:")
    for num, (name, data) in SCENARIOS.items():
        ingredient_count = len(data.get("ingredients", []))
        recipe_count = len(data.get("recipes", []))
        menu_count = len(data.get("user_menus", []))
        print(f"  [{num}] {name}")
        print(f"      Ingredients: {ingredient_count}, Recipes: {recipe_count}, User Menus: {menu_count}")

    while True:
        try:
            choice = input("\nSelect scenario number (or 0 to cancel): ").strip()
            if choice == "0":
                return None
            if int(choice) in SCENARIOS:
                scenario_num = int(choice)
                name, data = SCENARIOS[scenario_num]
                db = KitchenDB()
                db.init_from_json(data)
                print(f"\nInitialized: {name}")
                print(f"  Ingredients loaded: {len(db.ingredients)}")
                print(f"  Recipes loaded: {len(db.recipes)}")
                return db
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def show_catalog_summary(db: KitchenDB):
    """Show a summary of the catalog."""
    print("\nCatalog Summary:")
    print(f"  Total ingredients: {len(db.ingredients)}")
    print(f"  Total recipes: {len(db.recipes)}")
    print(f"  User menus: {len(db.user_menus)}")
    print(f"  Shopping lists: {len(db.user_shopping_lists)}")

    # Count by category
    categories = {}
    for ing in db.ingredients.values():
        cat = ing.category
        categories[cat] = categories.get(cat, 0) + 1

    print("\n  Ingredient Categories:")
    for cat, count in sorted(categories.items()):
        print(f"    - {cat}: {count}")

    # Recipes by difficulty
    difficulties = {}
    for recipe in db.recipes.values():
        diff = getattr(recipe, 'difficulty', 'N/A')
        difficulties[diff] = difficulties.get(diff, 0) + 1

    print("\n  Recipes by Difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"    - {diff}: {count}")

    # Taste profiles
    tastes = {}
    for recipe in db.recipes.values():
        for taste in recipe.taste:
            tastes[taste] = tastes.get(taste, 0) + 1

    print("\n  Recipes by Taste:")
    for taste, count in sorted(tastes.items()):
        print(f"    - {taste}: {count}")


def test_get_all_ingredients(db: KitchenDB):
    """Test get_all_ingredient_names method."""
    result = db.get_all_ingredient_names()
    print("\nResult:")
    print(f"  Total ingredients: {len(result.get('ingredients', []))}")
    for ing in result.get("ingredients", [])[:30]:
        print(f"    - {ing}")
    if len(result.get("ingredients", [])) > 30:
        print(f"    ... and {len(result['ingredients']) - 30} more")


def test_get_ingredients_by_category(db: KitchenDB):
    """Test get_ingredients_by_category method."""
    category = input("Enter category (e.g., vegetable, meat, dairy): ").strip()
    result = db.get_ingredients_by_category(category)
    print("\nResult:")
    print(f"  Ingredients found: {len(result.get('ingredients', []))}")
    for ing in result.get("ingredients", [])[:20]:
        print(f"    - {ing}")


def test_find_ingredient_category(db: KitchenDB):
    """Test find_ingredient_category method."""
    ingredient_name = input("Enter ingredient name: ").strip()
    result = db.find_ingredient_category(ingredient_name)
    print(f"\nResult: {result}")


def test_get_ingredient_nutrition(db: KitchenDB):
    """Test get_ingredient_nutrition method."""
    ingredient_name = input("Enter ingredient name: ").strip()
    result = db.get_ingredient_nutrition(ingredient_name)
    print("\nResult:")
    if "nutrition" in result:
        print(f"  Ingredient: {result.get('ingredient_name', 'N/A')}")
        for key, value in result["nutrition"].items():
            if value is not None:
                print(f"    {key}: {value}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_ingredient_quantity(db: KitchenDB):
    """Test get_ingredient_quantity method."""
    ingredient_name = input("Enter ingredient name: ").strip()
    result = db.get_ingredient_quantity(ingredient_name)
    print(f"\nResult: {result}")


def test_get_ingredient_location(db: KitchenDB):
    """Test get_ingredient_location method."""
    ingredient_name = input("Enter ingredient name: ").strip()
    result = db.get_ingredient_location(ingredient_name)
    print(f"\nResult: {result}")


def test_find_ingredients_by_location(db: KitchenDB):
    """Test find_ingredients_by_location method."""
    location = input("Enter storage location (e.g., fridge, pantry): ").strip()
    result = db.find_ingredients_by_location(location)
    print("\nResult:")
    print(f"  Ingredients found: {len(result.get('ingredients', []))}")
    for ing in result.get("ingredients", [])[:20]:
        print(f"    - {ing}")


def test_get_ingredient_shelf_life(db: KitchenDB):
    """Test get_ingredient_shelf_life method."""
    ingredient_name = input("Enter ingredient name: ").strip()
    result = db.get_ingredient_shelf_life(ingredient_name)
    print(f"\nResult: {result}")


def test_find_ingredients_by_expiry_date(db: KitchenDB):
    """Test find_ingredients_by_expiry_date method."""
    date = input("Enter date (YYYY-MM-DD): ").strip()
    result = db.find_ingredients_by_expiry_date(date)
    print("\nResult:")
    print(f"  Ingredients found: {len(result.get('ingredients', []))}")
    for ing in result.get("ingredients", [])[:20]:
        print(f"    - {ing}")


def test_get_all_recipes(db: KitchenDB):
    """Test get_all_recipe_names method."""
    result = db.get_all_recipe_names()
    print("\nResult:")
    print(f"  Total recipes: {len(result.get('recipes', []))}")
    for recipe in result.get("recipes", [])[:30]:
        print(f"    - {recipe}")
    if len(result.get("recipes", [])) > 30:
        print(f"    ... and {len(result['recipes']) - 30} more")


def test_search_recipe(db: KitchenDB):
    """Test searching for a recipe (using get_cooking_steps as search)."""
    recipe_name = input("Enter recipe name: ").strip()

    # First try to get recipe info
    print("\n--- Recipe Info ---")
    result = db.get_recipe_ingredients(recipe_name)
    if result.get("ingredients"):
        print(f"  Recipe: {result.get('recipe_name', 'N/A')}")
        print("  Ingredients:")
        for ing in result["ingredients"]:
            print(f"    - {ing['ingredient_name']}: {ing['quantity']}")

        # Also get cooking steps
        steps_result = db.get_cooking_steps(recipe_name)
        if steps_result.get("steps"):
            print("\n  Cooking Steps:")
            for step in steps_result["steps"]:
                print(f"    {step['step_number']}. {step['description']}")
    else:
        print(f"  Recipe not found: {recipe_name}")


def test_get_recipe_ingredients(db: KitchenDB):
    """Test get_recipe_ingredients method."""
    recipe_name = input("Enter recipe name: ").strip()
    result = db.get_recipe_ingredients(recipe_name)
    print("\nResult:")
    if "ingredients" in result:
        print(f"  Recipe: {result.get('recipe_name', 'N/A')}")
        print("  Required ingredients:")
        for ing in result["ingredients"]:
            print(f"    - {ing.get('ingredient_name', 'N/A')}: {ing.get('quantity', 'N/A')}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_cooking_steps(db: KitchenDB):
    """Test get_cooking_steps method."""
    recipe_name = input("Enter recipe name: ").strip()
    result = db.get_cooking_steps(recipe_name)
    print("\nResult:")
    if "steps" in result:
        print(f"  Recipe: {result.get('recipe_name', 'N/A')}")
        print("  Cooking Steps:")
        for step in result["steps"]:
            print(f"    {step['step_number']}. {step['description']}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_recipe_allergens(db: KitchenDB):
    """Test get_recipe_allergens method."""
    recipe_name = input("Enter recipe name: ").strip()
    result = db.get_recipe_allergens(recipe_name)
    print(f"\nResult: {result}")


def test_find_recipes_by_allergen(db: KitchenDB):
    """Test find_recipes_by_allergen method."""
    allergen = input("Enter allergen (e.g., dairy, nuts, gluten): ").strip()
    result = db.find_recipes_by_allergen(allergen)
    print("\nResult:")
    print(f"  Recipes found: {len(result.get('recipes', []))}")
    for recipe in result.get("recipes", [])[:20]:
        print(f"    - {recipe}")


def test_get_recipe_taste(db: KitchenDB):
    """Test get_recipe_taste method."""
    recipe_name = input("Enter recipe name: ").strip()
    result = db.get_recipe_taste(recipe_name)
    print(f"\nResult: {result}")


def test_find_recipes_by_taste(db: KitchenDB):
    """Test find_recipes_by_taste method."""
    taste = input("Enter taste profile (e.g., savory, sweet, spicy): ").strip()
    result = db.find_recipes_by_taste(taste)
    print("\nResult:")
    print(f"  Recipes found: {len(result.get('recipes', []))}")
    for recipe in result.get("recipes", [])[:20]:
        print(f"    - {recipe}")


def test_get_recipe_nutritional_characteristics(db: KitchenDB):
    """Test get_recipe_nutritional_characteristics method."""
    recipe_name = input("Enter recipe name: ").strip()
    result = db.get_recipe_nutritional_characteristics(recipe_name)
    print(f"\nResult: {result}")


def test_find_recipes_by_nutritional_characteristics(db: KitchenDB):
    """Test find_recipes_by_nutritional_characteristics method."""
    characteristic = input("Enter nutritional characteristic (e.g., high_protein, low_fat): ").strip()
    result = db.find_recipes_by_nutritional_characteristics(characteristic)
    print("\nResult:")
    print(f"  Recipes found: {len(result.get('recipes', []))}")
    for recipe in result.get("recipes", [])[:20]:
        print(f"    - {recipe}")


def test_find_recipes_by_ingredient(db: KitchenDB):
    """Test find_recipes_by_ingredient method."""
    ingredient = input("Enter ingredient name: ").strip()
    result = db.find_recipes_by_ingredient(ingredient)
    print("\nResult:")
    print(f"  Recipes found: {len(result.get('recipes', []))}")
    for recipe in result.get("recipes", [])[:20]:
        print(f"    - {recipe}")


def test_add_recipe_to_menu(db: KitchenDB):
    """Test add_recipe_to_menu method."""
    user_id = input("Enter user ID: ").strip()
    recipe_name = input("Enter recipe name: ").strip()
    result = db.add_recipe_to_menu(user_id, recipe_name)
    print(f"\nResult: {result}")


def test_remove_recipe_from_menu(db: KitchenDB):
    """Test remove_recipe_from_menu method."""
    user_id = input("Enter user ID: ").strip()
    recipe_name = input("Enter recipe name: ").strip()
    result = db.remove_recipe_from_menu(user_id, recipe_name)
    print(f"\nResult: {result}")


def test_get_current_menu(db: KitchenDB):
    """Test get_current_menu method."""
    user_id = input("Enter user ID: ").strip()
    result = db.get_current_menu(user_id)
    print("\nResult:")
    print(f"  User: {result.get('user_id', 'N/A')}")
    print(f"  Recipes: {result.get('recipes', [])}")


def test_tally_nutritional_characteristics(db: KitchenDB):
    """Test tally_total_nutritional_characteristics method."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter recipe names (one per line, empty line to finish):")

    recipes = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        recipes.append(line)

    if not recipes:
        print("No recipes entered.")
        return

    result = db.tally_total_nutritional_characteristics(user_id, recipes)
    print(f"\nResult:")
    print(f"  Nutritional characteristics: {result.get('nutritional_characteristics', [])}")


def test_tally_tastes(db: KitchenDB):
    """Test tally_total_tastes method."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter recipe names (one per line, empty line to finish):")

    recipes = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        recipes.append(line)

    if not recipes:
        print("No recipes entered.")
        return

    result = db.tally_total_tastes(user_id, recipes)
    print(f"\nResult:")
    print(f"  Tastes: {result.get('tastes', [])}")


def test_add_to_shopping_list(db: KitchenDB):
    """Test add_to_shopping_list method."""
    user_id = input("Enter user ID: ").strip()
    ingredient_name = input("Enter ingredient name: ").strip()
    try:
        quantity = float(input("Enter quantity: ").strip())
        result = db.add_to_shopping_list(user_id, ingredient_name, quantity)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_remove_from_shopping_list(db: KitchenDB):
    """Test remove_from_shopping_list method."""
    user_id = input("Enter user ID: ").strip()
    ingredient_name = input("Enter ingredient name: ").strip()
    result = db.remove_from_shopping_list(user_id, ingredient_name)
    print(f"\nResult: {result}")


def test_get_shopping_list(db: KitchenDB):
    """Test get_current_shopping_list method."""
    user_id = input("Enter user ID: ").strip()
    result = db.get_current_shopping_list(user_id)
    print("\nResult:")
    print(f"  User: {result.get('user_id', 'N/A')}")
    if result.get("items"):
        print("  Shopping list:")
        for item in result["items"]:
            print(f"    - {item.get('ingredient_name', 'N/A')}: {item.get('quantity', 'N/A')}")
    else:
        print("  Shopping list is empty")


def test_compute_total_nutrition(db: KitchenDB):
    """Test compute_total_nutritions method."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter ingredients (one per line, format: ingredient_name,quantity)")
    print("Enter empty line when done:")

    ingredients = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                ingredients.append({
                    "ingredient_name": parts[0].strip(),
                    "quantity": float(parts[1].strip())
                })
            except ValueError:
                print("  Invalid format. Use: ingredient_name,quantity")
        else:
            print("  Invalid format. Use: ingredient_name,quantity")

    if not ingredients:
        print("No ingredients entered.")
        return

    result = db.compute_total_nutritions(user_id, ingredients)
    print(f"\nResult:")
    if "total_nutrition" in result:
        print("  Total Nutrition:")
        for key, value in result["total_nutrition"].items():
            print(f"    {key}: {value}")
    else:
        print("  No nutrition data available.")


def main():
    """Main interactive loop."""
    db = None

    while True:
        if db is None:
            print("\n[!] Database not initialized. Please initialize first.")
            db = initialize_database()
            if db is None:
                print("Initialization cancelled.")
                break
            continue

        print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            db = initialize_database()
        elif choice == "2":
            show_catalog_summary(db)
        elif choice == "3":
            test_get_all_ingredients(db)
        elif choice == "4":
            test_get_ingredients_by_category(db)
        elif choice == "5":
            test_find_ingredient_category(db)
        elif choice == "6":
            test_get_ingredient_nutrition(db)
        elif choice == "7":
            test_get_ingredient_quantity(db)
        elif choice == "8":
            test_get_ingredient_location(db)
        elif choice == "9":
            test_find_ingredients_by_location(db)
        elif choice == "10":
            test_get_ingredient_shelf_life(db)
        elif choice == "11":
            test_find_ingredients_by_expiry_date(db)
        elif choice == "12":
            test_get_all_recipes(db)
        elif choice == "13":
            test_search_recipe(db)
        elif choice == "14":
            test_get_recipe_ingredients(db)
        elif choice == "15":
            test_get_cooking_steps(db)
        elif choice == "16":
            test_get_recipe_allergens(db)
        elif choice == "17":
            test_find_recipes_by_allergen(db)
        elif choice == "18":
            test_get_recipe_taste(db)
        elif choice == "19":
            test_find_recipes_by_taste(db)
        elif choice == "20":
            test_get_recipe_nutritional_characteristics(db)
        elif choice == "21":
            test_find_recipes_by_nutritional_characteristics(db)
        elif choice == "22":
            test_find_recipes_by_ingredient(db)
        elif choice == "23":
            test_add_recipe_to_menu(db)
        elif choice == "24":
            test_remove_recipe_from_menu(db)
        elif choice == "25":
            test_get_current_menu(db)
        elif choice == "26":
            test_tally_nutritional_characteristics(db)
        elif choice == "27":
            test_tally_tastes(db)
        elif choice == "28":
            test_add_to_shopping_list(db)
        elif choice == "29":
            test_remove_from_shopping_list(db)
        elif choice == "30":
            test_get_shopping_list(db)
        elif choice == "31":
            test_compute_total_nutrition(db)
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
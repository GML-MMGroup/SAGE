#!/usr/bin/env python3
"""
Restaurant Database Interactive Test

Allows manual selection of initialization data and testing of all database operations.
"""

import sys
from typing import Dict, Any, List

from restaurant_db import RestaurantDB
from restaurant_init import restaurant_init_data, restaurant_init_data5


# Map scenario numbers to their data
SCENARIOS = {
    1: ("Restaurant Main Data", restaurant_init_data),
    5: ("Restaurant Scenario 5", restaurant_init_data5),
}


def print_menu():
    """Print the main menu."""
    print("\n" + "=" * 60)
    print("RESTAURANT DATABASE INTERACTIVE TEST")
    print("=" * 60)
    print("\n=== Catalog Management ===")
    print("[1] Initialize Database")
    print("[2] Add Dish to Catalog")
    print("[3] Remove Dish from Catalog")
    print("[4] Update Dish Price")
    print("[5] Update Dish Discount")
    print("[6] Show Catalog Summary")
    print("\n=== Dish Search ===")
    print("[7] Find Dishes by Category")
    print("[8] Find Dishes by Nutritional Tag")
    print("[9] Find Dishes by Taste")
    print("[11] List All Discounted Dishes")
    print("\n=== Dish Information ===")
    print("[12] Get Dish Nutrition")
    print("[13] Get Dish Allergens")
    print("[14] Get Dish Taste Profile")
    print("[15] Get Dish Price")
    print("[16] Get Dish Discount")
    print("\n=== Set Meal Management ===")
    print("[17] Create Set Meal")
    print("[18] Get Set Meal Details")
    print("[19] Find Set Meals Containing Dish")
    print("\n=== Order Management ===")
    print("[20] Add Dish to Order")
    print("[21] Remove Dish from Order")
    print("[22] Clear User Order")
    print("[23] Get User Order Summary")
    print("\n=== Calculations ===")
    print("[24] Compute Total Payment")
    print("[25] Compute Total Tax")
    print("[26] Compute Total Nutrition")
    print("[27] Add Set Meal to Order")
    print("\n[0] Exit")
    print("-" * 60)


def initialize_database() -> RestaurantDB:
    """Let user select and initialize a scenario."""
    print("\nAvailable Scenarios:")
    for num, (name, data) in SCENARIOS.items():
        dish_count = len(data.get("dishes", []))
        set_meal_count = len(data.get("set_meals", []))
        print(f"  [{num}] {name} ({dish_count} dishes, {set_meal_count} set meals)")

    while True:
        try:
            choice = input("\nSelect scenario number (or 0 to cancel): ").strip()
            if choice == "0":
                return None
            if int(choice) in SCENARIOS:
                scenario_num = int(choice)
                name, data = SCENARIOS[scenario_num]
                db = RestaurantDB()
                db.init_from_json(data)
                print(f"\nInitialized: {name}")
                print(f"  Dishes loaded: {len(db.catalog)}")
                print(f"  Set meals loaded: {len(db.set_meals)}")
                return db
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def test_add_dish_to_catalog(db: RestaurantDB):
    """Test add_dish_to_catalog method."""
    print("\n--- Add Dish to Catalog ---")
    name = input("Dish name: ").strip()
    category = input("Category (Pizza/Pasta/Salads/Sandwiches/Cold Cuts/Steaks/etc.): ").strip()
    try:
        price = float(input("Price: ").strip())
        tax_rate = float(input("Tax rate (e.g., 0.08): ").strip())
        discount = float(input("Discount (e.g., 1.0 for no discount): ").strip())
    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    nutritional_characteristics = input("Nutritional characteristics (comma-separated): ").strip().split(",")
    taste = input("Taste profile (comma-separated): ").strip().split(",")
    allergens = input("Allergens (comma-separated, empty for none): ").strip()
    allergen_list = [a.strip() for a in allergens.split(",")] if allergens else []

    print("\nNutrition Info:")
    try:
        nutrition = {
            "basis": input("  Basis (PER_100G/PER_SERVING): ").strip() or "PER_100G",
            "calories_kcal": float(input("  Calories: ").strip() or 0),
            "protein_g": float(input("  Protein (g): ").strip() or 0),
            "fat_g": float(input("  Fat (g): ").strip() or 0),
            "carbs_g": float(input("  Carbs (g): ").strip() or 0),
        }
    except ValueError as e:
        print(f"Invalid nutrition input: {e}")
        return

    result = db.add_dish_to_catalog(name, category, price, tax_rate, discount,
                                    nutritional_characteristics, taste, allergen_list, nutrition)
    print(f"\nResult: {result}")


def test_remove_dish_from_catalog(db: RestaurantDB):
    """Test remove_dish_from_catalog method."""
    name = input("Enter dish name to remove: ").strip()
    result = db.remove_dish_from_catalog(name)
    print(f"\nResult: {result}")


def test_update_dish_price(db: RestaurantDB):
    """Test update_dish_price method."""
    name = input("Enter dish name: ").strip()
    try:
        new_price = float(input("New price: ").strip())
        result = db.update_dish_price(name, new_price)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_update_dish_discount(db: RestaurantDB):
    """Test update_dish_discount method."""
    name = input("Enter dish name: ").strip()
    try:
        new_discount = float(input("New discount (0-1): ").strip())
        result = db.update_dish_discount(name, new_discount)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_find_dishes_by_category(db: RestaurantDB):
    """Test find_dishes_by_category method."""
    category = input("Enter category (e.g., Pizza, Cold Cuts, Pasta): ").strip()
    result = db.find_dishes_by_category(category)
    print("\nResult:")
    print(f"  Dishes found: {len(result.get('dishes', []))}")
    for dish in result.get("dishes", [])[:20]:
        print(f"    - {dish}")


def test_find_dishes_by_nutritional_tag(db: RestaurantDB):
    """Test find_dishes_by_nutritional_tag method."""
    tag = input("Enter nutritional tag (e.g., high_protein, low_fat): ").strip()
    result = db.find_dishes_by_nutritional_tag(tag)
    print("\nResult:")
    print(f"  Dishes found: {len(result.get('dishes', []))}")
    for dish in result.get("dishes", [])[:20]:
        print(f"    - {dish}")


def test_find_dishes_by_taste(db: RestaurantDB):
    """Test find_dishes_by_taste method."""
    taste = input("Enter taste profile (e.g., savory, sweet, salty): ").strip()
    result = db.find_dishes_by_taste(taste)
    print("\nResult:")
    print(f"  Dishes found: {len(result.get('dishes', []))}")
    for dish in result.get("dishes", [])[:20]:
        print(f"    - {dish}")


def test_list_all_discounted_dishes(db: RestaurantDB):
    """Test list_all_discounted_dishes method."""
    result = db.list_all_discounted_dishes()
    print("\nResult:")
    print(f"  Discounted dishes: {len(result.get('discounted_dishes', []))}")
    for dish in result.get("discounted_dishes", [])[:20]:
        print(f"    - {dish}")


def test_get_dish_nutrition(db: RestaurantDB):
    """Test get_dish_nutrition method."""
    dish_name = input("Enter dish name: ").strip()
    result = db.get_dish_nutrition(dish_name)
    print("\nResult:")
    if "matching_dishes" in result:
        for name, nutrition in result["matching_dishes"].items():
            print(f"  Dish: {name}")
            for key, value in nutrition.items():
                if value is not None:
                    print(f"    {key}: {value}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_dish_allergens(db: RestaurantDB):
    """Test get_dish_allergens method."""
    dish_name = input("Enter dish name: ").strip()
    result = db.get_dish_allergens(dish_name)
    print("\nResult:")
    if "matching_dishes" in result:
        for name, allergens in result["matching_dishes"].items():
            print(f"  - {name}: {allergens}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_dish_taste_profile(db: RestaurantDB):
    """Test get_dish_taste_profile method."""
    dish_name = input("Enter dish name: ").strip()
    result = db.get_dish_taste_profile(dish_name)
    print("\nResult:")
    if "matching_dishes" in result:
        for name, taste in result["matching_dishes"].items():
            print(f"  - {name}: {taste}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_dish_price(db: RestaurantDB):
    """Test get_dish_price method."""
    dish_name = input("Enter dish name: ").strip()
    result = db.get_dish_price(dish_name)
    print("\nResult:")
    if "matching_dishes" in result:
        for name, price in result["matching_dishes"].items():
            print(f"  - {name}: ${price:.2f}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_dish_discount(db: RestaurantDB):
    """Test get_dish_discount method."""
    dish_name = input("Enter dish name: ").strip()
    result = db.get_dish_discount(dish_name)
    print("\nResult:")
    if "matching_dishes" in result:
        for name, discount in result["matching_dishes"].items():
            print(f"  - {name}: {discount * 100}%")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_create_set_meal(db: RestaurantDB):
    """Test create_set_meal method."""
    set_meal_name = input("Enter set meal name: ").strip()
    print("\nEnter included dishes (one per line, format: dish_name,quantity)")
    print("Enter empty line when done:")

    included_dishes = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                included_dishes.append({
                    "dish_name": parts[0].strip(),
                    "quantity": float(parts[1].strip())
                })
            except ValueError:
                print("  Invalid format. Use: dish_name,quantity")
        else:
            print("  Invalid format. Use: dish_name,quantity")

    if not included_dishes:
        print("No dishes entered.")
        return

    try:
        set_meal_price = float(input("Set meal price: ").strip())
        set_meal_discount = float(input("Set meal discount (0-1): ").strip())

        result = db.create_set_meal(set_meal_name, included_dishes, set_meal_price, set_meal_discount)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_get_set_meal_details(db: RestaurantDB):
    """Test get_set_meal_details method."""
    set_meal_name = input("Enter set meal name: ").strip()
    result = db.get_set_meal_details(set_meal_name)
    print("\nResult:")
    if "name" in result:
        print(f"  Name: {result['name']}")
        print(f"  Price: ${result.get('price', 0):.2f}")
        print(f"  Discount: {result.get('discount', 1) * 100}%")
        print(f"  Included dishes:")
        for dish in result.get("included_dishes", []):
            print(f"    - {dish.get('dish_name', 'N/A')}: {dish.get('quantity', 1)}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_find_set_meals_containing_dish(db: RestaurantDB):
    """Test find_set_meals_containing_dish method."""
    dish_name = input("Enter dish name: ").strip()
    result = db.find_set_meals_containing_dish(dish_name)
    print("\nResult:")
    print(f"  Set meals found: {len(result.get('set_meals', []))}")
    for meal in result.get("set_meals", []):
        print(f"    - {meal}")


def test_add_dish_to_order(db: RestaurantDB):
    """Test add_dish_to_order method."""
    user_id = input("Enter user ID: ").strip()
    dish_name = input("Enter dish name: ").strip()
    try:
        quantity = int(input("Enter quantity: ").strip())
        result = db.add_dish_to_order(user_id, dish_name, quantity)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_remove_dish_from_order(db: RestaurantDB):
    """Test remove_dish_from_order method."""
    user_id = input("Enter user ID: ").strip()
    dish_name = input("Enter dish name to remove: ").strip()
    try:
        quantity = int(input("Enter quantity to remove: ").strip())
        result = db.remove_dish_from_order(user_id, dish_name, quantity)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_clear_user_order(db: RestaurantDB):
    """Test clear_user_order method."""
    user_id = input("Enter user ID: ").strip()
    result = db.clear_user_order(user_id)
    print(f"\nResult: {result}")


def test_get_user_order_summary(db: RestaurantDB):
    """Test get_user_order_summary method."""
    user_id = input("Enter user ID: ").strip()
    result = db.get_user_order_summary(user_id)
    print("\nResult:")
    print(f"  Total items: {result.get('total_items', 0)}")
    if result.get("items"):
        print("  Order contents:")
        for item in result["items"]:
            print(f"    - {item['dish_name']}: qty={item['quantity']}")
    else:
        print("  Order is empty")


def test_compute_total_payment(db: RestaurantDB):
    """Test compute_total_payment method."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter dishes (one per line, format: dish_name,quantity)")
    print("Enter empty line when done:")

    dishes = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                dishes.append({
                    "dish_name": parts[0].strip().lower(),
                    "quantity": float(parts[1].strip())
                })
            except ValueError:
                print("  Invalid format. Use: dish_name,quantity")

    if not dishes:
        print("No dishes entered.")
        return

    result = db.compute_total_payment(user_id, dishes)
    print(f"\nResult:")
    print(f"  Total: ${result.get('total', 0):.2f}")


def test_compute_total_tax(db: RestaurantDB):
    """Test compute_total_tax method."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter dishes (one per line, format: dish_name,quantity)")
    print("Enter empty line when done:")

    dishes = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                dishes.append({
                    "dish_name": parts[0].strip().lower(),
                    "quantity": float(parts[1].strip())
                })
            except ValueError:
                print("  Invalid format. Use: dish_name,quantity")

    if not dishes:
        print("No dishes entered.")
        return

    result = db.compute_total_tax(user_id, dishes)
    print(f"\nResult:")
    print(f"  Total tax: ${result.get('total_tax', 0):.2f}")


def test_compute_total_nutrition(db: RestaurantDB):
    """Test compute_total_nutrition method."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter dishes (one per line, format: dish_name,quantity)")
    print("Enter empty line when done:")

    dishes = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                dishes.append({
                    "dish_name": parts[0].strip().lower(),
                    "quantity": float(parts[1].strip())
                })
            except ValueError:
                print("  Invalid format. Use: dish_name,quantity")

    if not dishes:
        print("No dishes entered.")
        return

    result = db.compute_total_nutrition(user_id, dishes)
    print(f"\nResult:")
    if "total_nutrition" in result:
        print("  Total Nutrition:")
        for key, value in result["total_nutrition"].items():
            print(f"    {key}: {value}")


def test_add_set_meal_to_order(db: RestaurantDB):
    """Test add_set_meal_to_order method."""
    user_id = input("Enter user ID: ").strip()
    set_meal_name = input("Enter set meal name: ").strip()
    try:
        quantity = int(input("Enter quantity: ").strip())
        result = db.add_set_meal_to_order(user_id, set_meal_name, quantity)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def show_catalog_summary(db: RestaurantDB):
    """Show a summary of the catalog."""
    print("\nCatalog Summary:")
    print(f"  Total dishes: {len(db.catalog)}")
    print(f"  Total set meals: {len(db.set_meals)}")

    categories = {}
    for dish in db.catalog.values():
        cat = dish.category
        categories[cat] = categories.get(cat, 0) + 1

    print("\n  Categories:")
    for cat, count in sorted(categories.items()):
        print(f"    - {cat}: {count}")

    prices = [d.price for d in db.catalog.values()]
    if prices:
        print(f"\n  Price range: ${min(prices):.2f} - ${max(prices):.2f}")

    if db.set_meals:
        print("\n  Set Meals:")
        for name, meal in db.set_meals.items():
            print(f"    - {name}: ${meal.set_meal_price:.2f} ({len(meal.included_dishes)} dishes)")


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
            test_add_dish_to_catalog(db)
        elif choice == "3":
            test_remove_dish_from_catalog(db)
        elif choice == "4":
            test_update_dish_price(db)
        elif choice == "5":
            test_update_dish_discount(db)
        elif choice == "6":
            show_catalog_summary(db)
        elif choice == "7":
            test_find_dishes_by_category(db)
        elif choice == "8":
            test_find_dishes_by_nutritional_tag(db)
        elif choice == "9":
            test_find_dishes_by_taste(db)
        elif choice == "11":
            test_list_all_discounted_dishes(db)
        elif choice == "12":
            test_get_dish_nutrition(db)
        elif choice == "13":
            test_get_dish_allergens(db)
        elif choice == "14":
            test_get_dish_taste_profile(db)
        elif choice == "15":
            test_get_dish_price(db)
        elif choice == "16":
            test_get_dish_discount(db)
        elif choice == "17":
            test_create_set_meal(db)
        elif choice == "18":
            test_get_set_meal_details(db)
        elif choice == "19":
            test_find_set_meals_containing_dish(db)
        elif choice == "20":
            test_add_dish_to_order(db)
        elif choice == "21":
            test_remove_dish_from_order(db)
        elif choice == "22":
            test_clear_user_order(db)
        elif choice == "23":
            test_get_user_order_summary(db)
        elif choice == "24":
            test_compute_total_payment(db)
        elif choice == "25":
            test_compute_total_tax(db)
        elif choice == "26":
            test_compute_total_nutrition(db)
        elif choice == "27":
            test_add_set_meal_to_order(db)
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

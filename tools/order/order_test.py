#!/usr/bin/env python3
"""
Order Database Interactive Test

Allows manual selection of initialization data and testing of all database operations.
"""

import sys
from typing import Dict, Any, List

from order_db import OrderDB
from order_init import order_init_data


# Map scenario numbers to their data
SCENARIOS = {
    1: ("Order Main Data", order_init_data),
}


def print_menu():
    """Print the main menu."""
    print("\n" + "=" * 60)
    print("ORDER DATABASE INTERACTIVE TEST")
    print("=" * 60)
    print("\n=== Catalog Management ===")
    print("[1] Initialize Database")
    print("[2] Select Restaurant")
    print("[3] Add Dish to Catalog")
    print("[4] Remove Dish from Catalog")
    print("[5] Update Dish Price")
    print("[6] Update Dish Discount")
    print("[7] Show Catalog Summary")
    print("\n=== Dish Search ===")
    print("[8] Find Dishes by Category")
    print("[9] Find Dishes by Nutritional Tag")
    print("[10] Find Dishes by Taste")
    print("[12] List All Discounted Dishes")
    print("\n=== Dish Information ===")
    print("[13] Get Dish Nutrition")
    print("[14] Get Dish Allergens")
    print("[15] Get Dish Taste Profile")
    print("[16] Get Dish Price")
    print("[17] Get Dish Discount")
    print("\n=== Set Meal Management ===")
    print("[18] Create Set Meal")
    print("[19] Get Set Meal Details")
    print("[20] Find Set Meals Containing Dish")
    print("\n=== Order Management ===")
    print("[21] Add Dish to Order")
    print("[22] Remove Dish from Order")
    print("[23] Clear User Order")
    print("[24] Get User Order Summary")
    print("[25] Calculate Order Total")
    print("[26] Calculate Order Tax")
    print("[27] Summarize Order Nutrition")
    print("[28] Add Set Meal to Order")
    print("\n=== Multi-Restaurant Calculations ===")
    print("[29] Compute Total Payment (Multi-Restaurant)")
    print("[30] Compute Total Tax (Multi-Restaurant)")
    print("[31] Compute Total Nutrition (Multi-Restaurant)")
    print("\n[0] Exit")
    print("-" * 60)


def initialize_database() -> OrderDB:
    """Let user select and initialize a scenario."""
    print("\nAvailable Scenarios:")
    for num, (name, data) in SCENARIOS.items():
        dish_count = len(data.get("dishes", []))
        set_meal_count = len(data.get("set_meals", []))
        restaurant_count = len(set(d.get("restaurant_name", "Unknown") for d in data.get("dishes", [])))
        print(f"  [{num}] {name}")
        print(f"      Restaurants: {restaurant_count}, Dishes: {dish_count}, Set Meals: {set_meal_count}")

    while True:
        try:
            choice = input("\nSelect scenario number (or 0 to cancel): ").strip()
            if choice == "0":
                return None
            if int(choice) in SCENARIOS:
                scenario_num = int(choice)
                name, data = SCENARIOS[scenario_num]
                db = OrderDB()
                db.init_from_json(data)
                print(f"\nInitialized: {name}")
                print(f"  Restaurants: {len(db.restaurants)}")
                return db
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_current_restaurant(db: OrderDB) -> str:
    """Get current restaurant or prompt user to select one."""
    if db.user_current_restaurant:
        return db.user_current_restaurant

    print("\nNo active restaurant. Available restaurants:")
    for i, rest_name in enumerate(db.restaurants.keys(), 1):
        print(f"  [{i}] {rest_name}")

    try:
        choice = input("Select restaurant: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(db.restaurants):
            rest_list = list(db.restaurants.keys())
            return rest_list[int(choice) - 1]
        elif choice in db.restaurants:
            return choice
        else:
            print("Invalid choice.")
            return None
    except ValueError:
        print("Invalid input.")
        return None


def test_select_restaurant(db: OrderDB):
    """Test selecting a restaurant."""
    print("\nAvailable restaurants:")
    for i, rest_name in enumerate(db.restaurants.keys(), 1):
        print(f"  [{i}] {rest_name}")

    try:
        choice = input("Select restaurant: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(db.restaurants):
            rest_list = list(db.restaurants.keys())
            db.user_current_restaurant = rest_list[int(choice) - 1]
            print(f"Selected: {db.user_current_restaurant}")
        elif choice in db.restaurants:
            db.user_current_restaurant = choice
            print(f"Selected: {db.user_current_restaurant}")
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")


def test_add_dish_to_catalog(db: OrderDB):
    """Test add_dish_to_catalog method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    print(f"\n--- Add Dish to {restaurant} ---")
    name = input("Dish name: ").strip()
    category = input("Category: ").strip()
    try:
        price = float(input("Price: ").strip())
        tax_rate = float(input("Tax rate (e.g., 0.08): ").strip())
        discount = float(input("Discount (e.g., 1.0): ").strip())
    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    nutritional_characteristics = input("Nutritional characteristics (comma-separated): ").strip().split(",")
    taste = input("Taste profile (comma-separated): ").strip().split(",")
    allergens = input("Allergens (comma-separated): ").strip()
    allergen_list = [a.strip() for a in allergens.split(",")] if allergens else []

    print("\nNutrition Info:")
    try:
        nutrition = {
            "basis": input("  Basis: ").strip() or "PER_100G",
            "calories_kcal": float(input("  Calories: ").strip() or 0),
            "protein_g": float(input("  Protein (g): ").strip() or 0),
            "fat_g": float(input("  Fat (g): ").strip() or 0),
            "carbs_g": float(input("  Carbs (g): ").strip() or 0),
        }
    except ValueError as e:
        print(f"Invalid nutrition input: {e}")
        return

    result = db.add_dish_to_catalog(restaurant, name, category, price, tax_rate, discount,
                                    nutritional_characteristics, taste, allergen_list, nutrition)
    print(f"\nResult: {result}")


def test_remove_dish_from_catalog(db: OrderDB):
    """Test remove_dish_from_catalog method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    name = input("Enter dish name to remove: ").strip()
    result = db.remove_dish_from_catalog(restaurant, name)
    print(f"\nResult: {result}")


def test_update_dish_price(db: OrderDB):
    """Test update_dish_price method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    name = input("Enter dish name: ").strip()
    try:
        new_price = float(input("New price: ").strip())
        result = db.update_dish_price(restaurant, name, new_price)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_update_dish_discount(db: OrderDB):
    """Test update_dish_discount method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    name = input("Enter dish name: ").strip()
    try:
        new_discount = float(input("New discount (0-1): ").strip())
        result = db.update_dish_discount(restaurant, name, new_discount)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_find_dishes_by_category(db: OrderDB):
    """Test find_dishes_by_category method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    category = input("Enter category: ").strip()
    result = db.find_dishes_by_category(restaurant, category)
    print("\nResult:")
    print(f"  Dishes found: {len(result.get('dishes', []))}")
    for dish in result.get("dishes", [])[:20]:
        print(f"    - {dish}")


def test_find_dishes_by_nutritional_tag(db: OrderDB):
    """Test find_dishes_by_nutritional_tag method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    tag = input("Enter nutritional tag: ").strip()
    result = db.find_dishes_by_nutritional_tag(restaurant, tag)
    print("\nResult:")
    print(f"  Dishes found: {len(result.get('dishes', []))}")
    for dish in result.get("dishes", [])[:20]:
        print(f"    - {dish}")


def test_find_dishes_by_taste(db: OrderDB):
    """Test find_dishes_by_taste method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    taste = input("Enter taste profile: ").strip()
    result = db.find_dishes_by_taste(restaurant, taste)
    print("\nResult:")
    print(f"  Dishes found: {len(result.get('dishes', []))}")
    for dish in result.get("dishes", [])[:20]:
        print(f"    - {dish}")


def test_list_all_discounted_dishes(db: OrderDB):
    """Test list_all_discounted_dishes method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    result = db.list_all_discounted_dishes(restaurant)
    print("\nResult:")
    print(f"  Discounted dishes: {len(result.get('discounted_dishes', []))}")
    for dish in result.get("discounted_dishes", [])[:20]:
        print(f"    - {dish}")


def test_get_dish_nutrition(db: OrderDB):
    """Test get_dish_nutrition method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    dish_name = input("Enter dish name: ").strip()
    result = db.get_dish_nutrition(restaurant, dish_name)
    print("\nResult:")
    if "matching_dishes" in result:
        for name, nutrition in result["matching_dishes"].items():
            print(f"  Dish: {name}")
            for key, value in nutrition.items():
                if value is not None:
                    print(f"    {key}: {value}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_dish_allergens(db: OrderDB):
    """Test get_dish_allergens method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    dish_name = input("Enter dish name: ").strip()
    result = db.get_dish_allergens(restaurant, dish_name)
    print("\nResult:")
    if "matching_dishes" in result:
        for name, allergens in result["matching_dishes"].items():
            print(f"  - {name}: {allergens}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_dish_taste_profile(db: OrderDB):
    """Test get_dish_taste_profile method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    dish_name = input("Enter dish name: ").strip()
    result = db.get_dish_taste_profile(restaurant, dish_name)
    print("\nResult:")
    if "matching_dishes" in result:
        for name, taste in result["matching_dishes"].items():
            print(f"  - {name}: {taste}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_dish_price(db: OrderDB):
    """Test get_dish_price method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    dish_name = input("Enter dish name: ").strip()
    result = db.get_dish_price(restaurant, dish_name)
    print("\nResult:")
    if "matching_dishes" in result:
        for name, price in result["matching_dishes"].items():
            print(f"  - {name}: ${price:.2f}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_dish_discount(db: OrderDB):
    """Test get_dish_discount method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    dish_name = input("Enter dish name: ").strip()
    result = db.get_dish_discount(restaurant, dish_name)
    print("\nResult:")
    if "matching_dishes" in result:
        for name, discount in result["matching_dishes"].items():
            print(f"  - {name}: {discount * 100}%")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_create_set_meal(db: OrderDB):
    """Test create_set_meal method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

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
                print("  Invalid format.")
        else:
            print("  Invalid format.")

    if not included_dishes:
        print("No dishes entered.")
        return

    try:
        set_meal_price = float(input("Set meal price: ").strip())
        set_meal_discount = float(input("Set meal discount (0-1): ").strip())

        result = db.create_set_meal(restaurant, set_meal_name, included_dishes, set_meal_price, set_meal_discount)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_get_set_meal_details(db: OrderDB):
    """Test get_set_meal_details method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    set_meal_name = input("Enter set meal name: ").strip()
    result = db.get_set_meal_details(restaurant, set_meal_name)
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


def test_find_set_meals_containing_dish(db: OrderDB):
    """Test find_set_meals_containing_dish method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    dish_name = input("Enter dish name: ").strip()
    result = db.find_set_meals_containing_dish(restaurant, dish_name)
    print("\nResult:")
    print(f"  Set meals found: {len(result.get('set_meals', []))}")
    for meal in result.get("set_meals", []):
        print(f"    - {meal}")


def test_add_dish_to_order(db: OrderDB):
    """Test add_dish_to_order method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    user_id = input("Enter user ID: ").strip()
    dish_name = input("Enter dish name: ").strip()
    try:
        quantity = int(input("Enter quantity: ").strip())
        result = db.add_dish_to_order(restaurant, user_id, dish_name, quantity)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_remove_dish_from_order(db: OrderDB):
    """Test remove_dish_from_order method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    user_id = input("Enter user ID: ").strip()
    dish_name = input("Enter dish name to remove: ").strip()
    try:
        quantity = int(input("Enter quantity to remove: ").strip())
        result = db.remove_dish_from_order(restaurant, user_id, dish_name, quantity)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_clear_user_order(db: OrderDB):
    """Test clear_user_order method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    user_id = input("Enter user ID: ").strip()
    result = db.clear_user_order(restaurant, user_id)
    print(f"\nResult: {result}")


def test_get_user_order_summary(db: OrderDB):
    """Test get_user_order_summary method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    user_id = input("Enter user ID: ").strip()
    result = db.get_user_order_summary(restaurant, user_id)
    print("\nResult:")
    print(f"  Total items: {result.get('total_items', 0)}")
    if result.get("items"):
        print("  Order contents:")
        for item in result["items"]:
            print(f"    - {item['dish_name']}: qty={item['quantity']}")
    else:
        print("  Order is empty")


def test_calculate_order_total(db: OrderDB):
    """Test calculate_order_total method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    user_id = input("Enter user ID: ").strip()
    result = db.calculate_order_total(restaurant, user_id)
    print("\nResult:")
    print(f"  Total: ${result.get('total', 0):.2f}")


def test_calculate_order_tax(db: OrderDB):
    """Test calculate_order_tax method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    user_id = input("Enter user ID: ").strip()
    result = db.calculate_order_tax(restaurant, user_id)
    print("\nResult:")
    print(f"  Total tax: ${result.get('total_tax', 0):.2f}")


def test_summarize_order_nutrition(db: OrderDB):
    """Test summarize_order_nutrition method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    user_id = input("Enter user ID: ").strip()
    result = db.summarize_order_nutrition(restaurant, user_id)
    print("\nResult:")
    if "total_nutrition" in result:
        print("  Total Nutrition:")
        for key, value in result["total_nutrition"].items():
            print(f"    {key}: {value}")
    else:
        print("  No nutrition data available.")


def test_add_set_meal_to_order(db: OrderDB):
    """Test add_set_meal_to_order method."""
    restaurant = get_current_restaurant(db)
    if not restaurant:
        return

    user_id = input("Enter user ID: ").strip()
    set_meal_name = input("Enter set meal name: ").strip()
    try:
        quantity = int(input("Enter quantity: ").strip())
        result = db.add_set_meal_to_order(restaurant, user_id, set_meal_name, quantity)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_compute_total_payment(db: OrderDB):
    """Test compute_total_payment method (multi-restaurant)."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter dishes from different restaurants (one per line)")
    print("Format: restaurant_name,dish_name,quantity")
    print("Enter empty line when done:")

    dishes = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                dishes.append({
                    "restaurant_name": parts[0].strip().lower(),
                    "dish_name": parts[1].strip().lower(),
                    "quantity": float(parts[2].strip())
                })
            except ValueError:
                print("  Invalid format.")
        else:
            print("  Invalid format. Use: restaurant_name,dish_name,quantity")

    if not dishes:
        print("No dishes entered.")
        return

    result = db.compute_total_payment(user_id, dishes)
    print(f"\nResult:")
    print(f"  Total payment: ${result.get('total_payment', 0):.2f}")


def test_compute_total_tax(db: OrderDB):
    """Test compute_total_tax method (multi-restaurant)."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter dishes from different restaurants (one per line)")
    print("Format: restaurant_name,dish_name,quantity")
    print("Enter empty line when done:")

    dishes = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                dishes.append({
                    "restaurant_name": parts[0].strip().lower(),
                    "dish_name": parts[1].strip().lower(),
                    "quantity": float(parts[2].strip())
                })
            except ValueError:
                print("  Invalid format.")
        else:
            print("  Invalid format. Use: restaurant_name,dish_name,quantity")

    if not dishes:
        print("No dishes entered.")
        return

    result = db.compute_total_tax(user_id, dishes)
    print(f"\nResult:")
    print(f"  Total tax: ${result.get('total_tax', 0):.2f}")


def test_compute_total_nutrition(db: OrderDB):
    """Test compute_total_nutrition method (multi-restaurant)."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter dishes from different restaurants (one per line)")
    print("Format: restaurant_name,dish_name,quantity")
    print("Enter empty line when done:")

    dishes = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                dishes.append({
                    "restaurant_name": parts[0].strip().lower(),
                    "dish_name": parts[1].strip().lower(),
                    "quantity": float(parts[2].strip())
                })
            except ValueError:
                print("  Invalid format.")
        else:
            print("  Invalid format. Use: restaurant_name,dish_name,quantity")

    if not dishes:
        print("No dishes entered.")
        return

    result = db.compute_total_nutrition(user_id, dishes)
    print(f"\nResult:")
    if "total_nutrition" in result:
        print("  Total Nutrition:")
        for key, value in result["total_nutrition"].items():
            print(f"    {key}: {value}")
    else:
        print("  No nutrition data available.")


def show_catalog_summary(db: OrderDB):
    """Show a summary of all restaurants."""
    print("\nCatalog Summary:")
    print(f"  Total restaurants: {len(db.restaurants)}")

    for rest_name, rest_data in db.restaurants.items():
        print(f"\n  [{rest_name}]")
        catalog = rest_data.get('catalog', {})
        set_meals = rest_data.get('set_meals', {})

        print(f"    Total dishes: {len(catalog)}")

        categories = {}
        for dish in catalog.values():
            cat = dish.category
            categories[cat] = categories.get(cat, 0) + 1

        print("    Categories:")
        for cat, count in sorted(categories.items()):
            print(f"      - {cat}: {count}")

        if catalog:
            prices = [d.price for d in catalog.values()]
            print(f"    Price range: ${min(prices):.2f} - ${max(prices):.2f}")

        if set_meals:
            print(f"    Set meals: {len(set_meals)}")
            for name, meal in set_meals.items():
                print(f"      - {name}: ${meal.set_meal_price:.2f}")


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
            test_select_restaurant(db)
        elif choice == "3":
            test_add_dish_to_catalog(db)
        elif choice == "4":
            test_remove_dish_from_catalog(db)
        elif choice == "5":
            test_update_dish_price(db)
        elif choice == "6":
            test_update_dish_discount(db)
        elif choice == "7":
            show_catalog_summary(db)
        elif choice == "8":
            test_find_dishes_by_category(db)
        elif choice == "9":
            test_find_dishes_by_nutritional_tag(db)
        elif choice == "10":
            test_find_dishes_by_taste(db)
        elif choice == "12":
            test_list_all_discounted_dishes(db)
        elif choice == "13":
            test_get_dish_nutrition(db)
        elif choice == "14":
            test_get_dish_allergens(db)
        elif choice == "15":
            test_get_dish_taste_profile(db)
        elif choice == "16":
            test_get_dish_price(db)
        elif choice == "17":
            test_get_dish_discount(db)
        elif choice == "18":
            test_create_set_meal(db)
        elif choice == "19":
            test_get_set_meal_details(db)
        elif choice == "20":
            test_find_set_meals_containing_dish(db)
        elif choice == "21":
            test_add_dish_to_order(db)
        elif choice == "22":
            test_remove_dish_from_order(db)
        elif choice == "23":
            test_clear_user_order(db)
        elif choice == "24":
            test_get_user_order_summary(db)
        elif choice == "25":
            test_calculate_order_total(db)
        elif choice == "26":
            test_calculate_order_tax(db)
        elif choice == "27":
            test_summarize_order_nutrition(db)
        elif choice == "28":
            test_add_set_meal_to_order(db)
        elif choice == "29":
            test_compute_total_payment(db)
        elif choice == "30":
            test_compute_total_tax(db)
        elif choice == "31":
            test_compute_total_nutrition(db)
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

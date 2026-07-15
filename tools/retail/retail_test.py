#!/usr/bin/env python3
"""
Retail Database Interactive Test

Allows manual selection of initialization data and testing of all database operations.
"""

import sys
from typing import Dict, Any, List

from retail_db import RetailDB
from retail_init import (
    retail_init_data1,
    retail_init_data2,
    retail_init_data3,
    retail_init_data4,
    retail_init_data5,
    retail_init_data6,
    retail_init_data7,
    retail_init_data8,
    retail_init_data9,
    retail_init_data10,
)


# Map scenario numbers to their data
SCENARIOS = {
    1: ("Retail Scenario 1", retail_init_data1),
    2: ("Retail Scenario 2", retail_init_data2),
    3: ("Retail Scenario 3", retail_init_data3),
    4: ("Retail Scenario 4", retail_init_data4),
    5: ("Retail Scenario 5", retail_init_data5),
    6: ("Retail Scenario 6", retail_init_data6),
    7: ("Retail Scenario 7", retail_init_data7),
    8: ("Retail Scenario 8", retail_init_data8),
    9: ("Retail Scenario 9", retail_init_data9),
    10: ("Retail Scenario 10", retail_init_data10),
}


def print_menu():
    """Print the main menu."""
    print("\n" + "=" * 60)
    print("RETAIL DATABASE INTERACTIVE TEST")
    print("=" * 60)
    print("\n=== Catalog Management ===")
    print("[1] Initialize Database")
    print("[2] Add Product")
    print("[3] Delete Product")
    print("[4] Show Catalog Summary")
    print("\n=== Product Search ===")
    print("[5] Search Product (get_price)")
    print("[6] Find Products by Price Range")
    print("[7] Find Products by Nutritional Characteristic")
    print("[8] Find Products by Taste")
    print("[9] Find Products by Country of Origin")
    print("[10] List Discounted Products")
    print("[24] Find Products by Allergen")
    print("[25] Find Products by Category")
    print("\n=== Product Information ===")
    print("[11] Get Nutrition Info")
    print("[12] Get Tax Rate")
    print("[13] Get Discount")
    print("[14] Get Category")
    print("\n=== Cart Management ===")
    print("[15] Add to Cart")
    print("[16] Get Cart Contents")
    print("[17] Remove from Cart")
    print("[18] Clear Cart")
    print("\n=== Calculations ===")
    print("[19] Compute Total Payment")
    print("[20] Compute Total Tax")
    print("[21] Compute Total Nutrition")
    print("[22] Get Shopping List")
    print("\n[0] Exit")
    print("-" * 60)


def initialize_database() -> RetailDB:
    """Let user select and initialize a scenario."""
    print("\nAvailable Scenarios:")
    for num, (name, data) in SCENARIOS.items():
        product_count = len(data.get("products", []))
        print(f"  [{num}] {name} ({product_count} products)")

    while True:
        try:
            choice = input("\nSelect scenario number (or 0 to cancel): ").strip()
            if choice == "0":
                return None
            if int(choice) in SCENARIOS:
                scenario_num = int(choice)
                name, data = SCENARIOS[scenario_num]
                db = RetailDB()
                db.init_from_json(data)
                print(f"\nInitialized: {name}")
                print(f"  Products loaded: {len(db.catalog)}")
                return db
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def test_add_product(db: RetailDB):
    """Test add_product method."""
    print("\n--- Add Product ---")
    name = input("Product name: ").strip()
    category = input("Category: ").strip()
    try:
        price = float(input("Price: ").strip())
        tax_rate = float(input("Tax rate (e.g., 0.08): ").strip())
        discount = float(input("Discount (e.g., 1.0 for no discount): ").strip())
    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    nutritional_characteristics = input("Nutritional characteristics (comma-separated): ").strip().split(",")
    taste = input("Taste profile (comma-separated): ").strip().split(",")
    country = input("Country of origin: ").strip()

    # Nutrition info
    print("\nNutrition Info (PER_100G or PER_SERVING):")
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

    allergens = input("Allergens (comma-separated, empty for none): ").strip()
    allergen_list = [a.strip() for a in allergens.split(",")] if allergens else []

    result = db.add_product(name, category, price, tax_rate, discount,
                           nutritional_characteristics, taste, country, nutrition, allergen_list)
    print(f"\nResult: {result}")


def test_delete_product(db: RetailDB):
    """Test delete_product method."""
    name = input("Enter product name to delete: ").strip()
    result = db.delete_product(name)
    print(f"\nResult: {result}")


def test_get_price(db: RetailDB):
    """Test get_price method."""
    query = input("Enter product name to search: ").strip()
    result = db.get_price(query)
    print("\nResult:")
    if "products" in result:
        for p in result["products"]:
            print(f"  - {p['product_name']}: ${p['price']}")
        print(f"  Count: {result.get('count', 0)}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_find_products_by_price_range(db: RetailDB):
    """Test find_products_by_price_range method."""
    try:
        min_price = float(input("Enter minimum price: ").strip())
        max_price = float(input("Enter maximum price: ").strip())
        result = db.find_products_by_price_range(min_price, max_price)
        print("\nResult:")
        print(f"  Products found: {len(result.get('product_names', []))}")
        for name in result.get("product_names", []):
            print(f"    - {name}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_find_products_by_nutritional_characteristic(db: RetailDB):
    """Test find_products_by_nutritional_characteristic method."""
    characteristic = input("Enter nutritional characteristic (e.g., low_fat, high_protein): ").strip()
    result = db.find_products_by_nutritional_characteristic(characteristic)
    print("\nResult:")
    print(f"  Products found: {len(result.get('product_names', []))}")
    for name in result.get("product_names", []):
        print(f"    - {name}")


def test_find_products_by_taste(db: RetailDB):
    """Test find_products_by_taste method."""
    taste = input("Enter taste profile (e.g., sweet, savory): ").strip()
    result = db.find_products_by_taste(taste)
    print("\nResult:")
    print(f"  Products found: {len(result.get('product_names', []))}")
    for name in result.get("product_names", []):
        print(f"    - {name}")


def test_find_products_by_country_of_origin(db: RetailDB):
    """Test find_products_by_country_of_origin method."""
    country = input("Enter country of origin: ").strip()
    result = db.find_products_by_country_of_origin(country)
    print("\nResult:")
    print(f"  Products found: {len(result.get('product_names', []))}")
    for name in result.get("product_names", []):
        print(f"    - {name}")


def test_find_products_by_category(db: RetailDB):
    """Test find_products_by_category method."""
    category = input("Enter category: ").strip()
    result = db.find_products_by_category(category)
    print("\nResult:")
    print(f"  Products found: {len(result.get('product_names', []))}")
    for name in result.get("product_names", []):
        print(f"    - {name}")


def test_list_discounted_products(db: RetailDB):
    """Test list_discounted_products method."""
    result = db.list_discounted_products()
    print("\nResult:")
    print(f"  Discounted products: {len(result.get('product_names', []))}")
    for name in result.get("product_names", []):
        print(f"    - {name}")


def test_get_nutrition(db: RetailDB):
    """Test get_nutrition method."""
    product_name = input("Enter product name: ").strip()
    result = db.get_nutrition(product_name)
    print("\nResult:")
    if "products" in result:
        for p in result["products"]:
            print(f"  Product: {p['product_name']}")
            nutrition = p.get("nutrition", {})
            for key, value in nutrition.items():
                if value is not None:
                    print(f"    {key}: {value}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_tax_rate(db: RetailDB):
    """Test get_tax_rate method."""
    product_name = input("Enter product name: ").strip()
    result = db.get_tax_rate(product_name)
    print("\nResult:")
    if "products" in result:
        for p in result["products"]:
            print(f"  - {p['product_name']}: {p['tax_rate'] * 100}%")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_discount(db: RetailDB):
    """Test get_discount method."""
    product_name = input("Enter product name: ").strip()
    result = db.get_discount(product_name)
    print("\nResult:")
    if "products" in result:
        for p in result["products"]:
            print(f"  - {p['product_name']}: {p['discount'] * 100}%")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_get_category(db: RetailDB):
    """Test get_category method."""
    product_name = input("Enter product name: ").strip()
    result = db.get_category(product_name)
    print("\nResult:")
    if "products" in result:
        for p in result["products"]:
            print(f"  - {p['product_name']}: {p['category']}")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_add_to_cart(db: RetailDB):
    """Test add_to_cart method."""
    user_id = input("Enter user ID: ").strip()
    product_name = input("Enter product name: ").strip()
    try:
        quantity = int(input("Enter quantity: ").strip())

        # Get product info from catalog
        product = db.catalog.get(product_name.lower())
        if not product:
            matches = db._find_matching_products(product_name)
            if matches:
                product = matches[0]
                print(f"Using product: {product.name}")
            else:
                print("Product not found in catalog.")
                return

        result = db.add_to_cart(
            user_id, product.name, quantity,
            product.category, product.price, product.tax_rate, product.discount
        )
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_get_cart(db: RetailDB):
    """Test get_cart method."""
    user_id = input("Enter user ID: ").strip()
    result = db.get_cart(user_id)
    print("\nResult:")
    print(f"  Total items: {len(result.get('cart_items', []))}")
    if result.get("cart_items"):
        print("  Cart contents:")
        for item in result["cart_items"]:
            print(f"    - {item['product_name']}: qty={item['quantity']}, price=${item['price']}")
    else:
        print("  Cart is empty")


def test_remove_from_cart(db: RetailDB):
    """Test remove_from_cart method."""
    user_id = input("Enter user ID: ").strip()
    product_name = input("Enter product name to remove: ").strip()
    try:
        quantity = float(input("Enter quantity to remove: ").strip())
        result = db.remove_from_cart(user_id, product_name, quantity)
        print(f"\nResult: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")


def test_clear_cart(db: RetailDB):
    """Test clear_cart method."""
    user_id = input("Enter user ID: ").strip()
    result = db.clear_cart(user_id)
    print(f"\nResult: {result}")


def test_compute_total_payment(db: RetailDB):
    """Test compute_total_payment method."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter products (one per line, format: product_name,quantity)")
    print("Enter empty line when done:")

    products = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                products.append({
                    "product_name": parts[0].strip(),
                    "quantity": float(parts[1].strip())
                })
            except ValueError:
                print("  Invalid format. Use: product_name,quantity")
        else:
            print("  Invalid format. Use: product_name,quantity")

    if not products:
        print("No products entered.")
        return

    result = db.compute_total_payment(user_id, products)
    print(f"\nResult:")
    print(f"  Status: {result.get('status', 'N/A')}")
    print(f"  Total: ${result.get('total', 0):.2f}")
    if result.get("message"):
        print(f"  Message: {result['message']}")


def test_compute_total_tax(db: RetailDB):
    """Test compute_total_tax method."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter products (one per line, format: product_name,quantity)")
    print("Enter empty line when done:")

    products = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                products.append({
                    "product_name": parts[0].strip(),
                    "quantity": float(parts[1].strip())
                })
            except ValueError:
                print("  Invalid format. Use: product_name,quantity")

    if not products:
        print("No products entered.")
        return

    result = db.compute_total_tax(user_id, products)
    print(f"\nResult:")
    print(f"  Status: {result.get('status', 'N/A')}")
    print(f"  Total tax: ${result.get('total_tax', 0):.2f}")


def test_compute_total_nutrition(db: RetailDB):
    """Test compute_total_nutrition method."""
    user_id = input("Enter user ID: ").strip()
    print("\nEnter products (one per line, format: product_name,quantity)")
    print("Enter empty line when done:")

    products = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                products.append({
                    "product_name": parts[0].strip(),
                    "quantity": float(parts[1].strip())
                })
            except ValueError:
                print("  Invalid format. Use: product_name,quantity")

    if not products:
        print("No products entered.")
        return

    result = db.compute_total_nutrition(user_id, products)
    print(f"\nResult:")
    print(f"  Status: {result.get('status', 'N/A')}")
    if "total_nutrition" in result:
        print("  Total Nutrition:")
        for key, value in result["total_nutrition"].items():
            if value is not None:
                print(f"    {key}: {value}")


def test_get_shopping_list(db: RetailDB):
    """Test get_shopping_list method."""
    user_id = input("Enter user ID: ").strip()
    result = db.get_shopping_list(user_id)
    print("\nResult:")
    if "shopping_list" in result:
        if result["shopping_list"]:
            print("  Shopping list:")
            for item in result["shopping_list"]:
                if isinstance(item, dict):
                    print(f"    - {item.get('product_name', 'N/A')}: {item.get('quantity', 'N/A')}")
                else:
                    print(f"    - {item}")
        else:
            print("  Shopping list is empty")
    else:
        print(f"  {result.get('message', 'Unknown error')}")


def test_find_products_by_allergen(db: RetailDB):
    """Test find_products_by_allergen method."""
    allergen = input("Enter allergen (e.g. grapes, milk, gluten, soy, nuts): ").strip()
    result = db.find_products_by_allergen(allergen)
    print("\nResult:")
    names = result.get("product_names", [])
    if names:
        print(f"  Found {len(names)} product(s) containing '{allergen}':")
        for name in names:
            print(f"    - {name}")
    else:
        print(f"  No products found containing '{allergen}'")


def show_catalog_summary(db: RetailDB):
    """Show a summary of the catalog."""
    print("\nCatalog Summary:")
    print(f"  Total products: {len(db.catalog)}")

    categories = {}
    for product in db.catalog.values():
        cat = product.category
        categories[cat] = categories.get(cat, 0) + 1

    print("  Categories:")
    for cat, count in sorted(categories.items()):
        print(f"    - {cat}: {count}")

    prices = [p.price for p in db.catalog.values()]
    if prices:
        print(f"  Price range: ${min(prices):.2f} - ${max(prices):.2f}")


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
            test_add_product(db)
        elif choice == "3":
            test_delete_product(db)
        elif choice == "4":
            show_catalog_summary(db)
        elif choice == "5":
            test_get_price(db)
        elif choice == "6":
            test_find_products_by_price_range(db)
        elif choice == "7":
            test_find_products_by_nutritional_characteristic(db)
        elif choice == "8":
            test_find_products_by_taste(db)
        elif choice == "9":
            test_find_products_by_country_of_origin(db)
        elif choice == "10":
            test_list_discounted_products(db)
        elif choice == "11":
            test_get_nutrition(db)
        elif choice == "12":
            test_get_tax_rate(db)
        elif choice == "13":
            test_get_discount(db)
        elif choice == "14":
            test_get_category(db)
        elif choice == "15":
            test_add_to_cart(db)
        elif choice == "16":
            test_get_cart(db)
        elif choice == "17":
            test_remove_from_cart(db)
        elif choice == "18":
            test_clear_cart(db)
        elif choice == "19":
            test_compute_total_payment(db)
        elif choice == "20":
            test_compute_total_tax(db)
        elif choice == "21":
            test_compute_total_nutrition(db)
        elif choice == "22":
            test_get_shopping_list(db)
        elif choice == "24":
            test_find_products_by_allergen(db)
        elif choice == "25":
            test_find_products_by_category(db)
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

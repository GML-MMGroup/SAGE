import difflib
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
import json
import random

# --- Data Classes for Restaurant Schema ---
@dataclass
class NutritionInfo:
    """Nutrition facts for a dish."""
    basis: str  # "PER_100G" or "PER_SERVING"
    serving_size_g: Optional[float] = None
    calories_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    carbs_g: Optional[float] = None
    sugar_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    fiber_g: Optional[float] = None

@dataclass
class Dish:
    """Represents a dish in the restaurant catalog."""
    name: str
    category: str  # Pizza, Pasta, Salads, Sandwiches & Panini, Antipasti & Snacks, Cheese & Olives, Cold Cuts, Steaks, Handmade Bread
    price: float
    tax_rate: float
    discount: float
    nutritional_characteristics: List[str]  # low_calories, low_fat, low_sugar, low_sodium, no_additives, high_fiber, high_protein, high_calcium, sugar_free, gluten_free, vegan
    taste: List[str]  # sweet, salty, spicy, sour, bitter, umami, mild, savory
    allergens: List[str]
    nutrition: NutritionInfo

@dataclass
class SetMeal:
    """Represents a set meal combining multiple dishes."""
    name: str
    included_dishes: List[Dict[str, Any]]  # [{"dish_name": str, "quantity": float}]
    set_meal_price: float
    set_meal_discount: float

@dataclass
class OrderItem:
    """Represents an item in a user's order."""
    dish_name: str
    quantity: float

# --- Restaurant System Class ---
class RestaurantDB:
    """
    Manages a dish catalog and user orders for a restaurant.
    """
    def __init__(self):
        # Global dish catalog: dish_name -> Dish
        self.catalog: Dict[str, Dish] = {}
        # Set meals catalog: set_meal_name -> SetMeal
        self.set_meals: Dict[str, SetMeal] = {}
        # User orders: user_id -> dish_name -> OrderItem
        self.user_orders: Dict[str, Dict[str, OrderItem]] = {}

    # --- Add General Fuzzy Matching Method ---
    def _find_matching_dishes(self, query: str) -> List[Dish]:
        """
        General method: exact match dish names first, then fuzzy match by bidirectional inclusion.
        Return exact dish if found; otherwise return all dishes where name contains query string, or query string contains dish name.
        """
        # Convert query string to lowercase, unify matching rules, avoid case sensitivity issues
        query_lower = query.lower().strip()
        if query_lower in self.catalog:
            return [self.catalog[query_lower]]

        matching_dishes = []
        
        # Iterate all dishes for bidirectional inclusion judgment
        for dish in self.catalog.values():
            dish_name_lower = dish.name.lower()
            # Core logic: Bidirectional inclusion match
            # Condition 1: Query string in dish name  Condition 2: Dish name in query string
            if query_lower in dish_name_lower or (len(dish_name_lower) > 1 and dish_name_lower in query_lower):
                matching_dishes.append(dish)
        
        return matching_dishes

    def _find_matching_set_meals(self, query: str) -> List[SetMeal]:
        """
        General method: Fuzzy match set meal names based on query string (bidirectional inclusion match)
        """
        query_lower = query.lower()
        matching_set_meals = []
        
        for set_meal in self.set_meals.values():
            set_meal_name_lower = set_meal.name.lower()
            if query_lower in set_meal_name_lower or set_meal_name_lower in query_lower:
                matching_set_meals.append(set_meal)
            
        return matching_set_meals

    def init_from_json(self, data: Dict[str, Any]) -> None:
        """
        Initialize the RestaurantDB from JSON data.

        Expected JSON structure:
        {
            "dishes": [
                {
                    "name": "Margherita Pizza",
                    "category": "Pizza",
                    "price": 12.99,
                    "tax_rate": 0.08,
                    "discount": 1.0,
                    "nutritional_characteristics": ["low_sodium", "gluten_free"],
                    "taste": ["savory", "mild"],
                    "allergens": ["dairy"],
                    "nutrition": {
                        "basis": "PER_SERVING",
                        "serving_size_g": 400.0,
                        "calories_kcal": 800.0,
                        "protein_g": 20.0,
                        "fat_g": 30.0,
                        "carbs_g": 90.0,
                        "sugar_g": 5.0,
                        "sodium_mg": 600.0,
                        "fiber_g": 4.0
                    }
                }
            ],
            "set_meals": [
                {
                    "name": "Family Feast",
                    "included_dishes": [
                        {"dish_name": "Margherita Pizza", "quantity": 1},
                        {"dish_name": "Caesar Salad", "quantity": 2},
                        {"dish_name": "Tiramisu", "quantity": 1}
                    ],
                    "set_meal_price": 39.99,
                    "set_meal_discount": 0.85
                }
            ],
            "user_orders": [
                {
                    "user_id": "customer_001",
                    "items": [
                        {
                            "dish_name": "Margherita Pizza",
                            "quantity": 2
                        },
                        {
                            "dish_name": "Caesar Salad",
                            "quantity": 1
                        }
                    ]
                }
            ]
        }
        """
        # Initialize dishes/catalog
        if "dishes" in data:
            dishes_data = data["dishes"]
            self.catalog.clear()
            # Iterate dish list
            for dish_info in dishes_data:
                # Extract dish name as dictionary key (convert to lowercase)
                dish_name = dish_info.get("name", " ").lower()
                if not dish_name:
                    print("Warning: Dish without name is skipped")
                    continue
                    
                # Process nutrition field
                nutrition_info = dish_info.get("nutrition", {})
                nutrition_obj = NutritionInfo(
                    basis=nutrition_info.get("basis"),
                    serving_size_g=nutrition_info.get("serving_size_g"),
                    calories_kcal=nutrition_info.get("calories_kcal"),
                    protein_g=nutrition_info.get("protein_g"),
                    fat_g=nutrition_info.get("fat_g"),
                    carbs_g=nutrition_info.get("carbs_g"),
                    sugar_g=nutrition_info.get("sugar_g"),
                    sodium_mg=nutrition_info.get("sodium_mg"),
                    fiber_g=nutrition_info.get("fiber_g")
                )

                dish = Dish(
                    name=dish_name,
                    category=dish_info.get("category", " ").lower(),
                    price=dish_info.get("price", 0.0),
                    tax_rate=dish_info.get("tax_rate", 0.0),
                    discount=dish_info.get("discount", 1.0),
                    nutritional_characteristics=[char.lower() for char in dish_info.get("nutritional_characteristics", [])],
                    taste=[t.lower() for t in dish_info.get("taste", [])],
                    allergens=[a.lower() for a in dish_info.get("allergens", [])],
                    nutrition=nutrition_obj
                )
                self.catalog[dish_name] = dish

        # Initialize set meals
        if "set_meals" in data:
            set_meals_data = data["set_meals"]
            self.set_meals.clear()
            # Iterate set meal list
            for set_meal_info in set_meals_data:
                set_meal_name = set_meal_info.get("name", " ").lower()
                if not set_meal_name:
                    print("Warning: Set meal without name is skipped")
                    continue
                    
                # Verify if dishes in set meal exist in catalog
                included_dishes = set_meal_info.get("included_dishes", [])
                valid_included_dishes = []
                for item in included_dishes:
                    dish_name = item.get("dish_name", " ").lower()
                    if dish_name in self.catalog:
                        valid_included_dishes.append({
                            "dish_name": dish_name,
                            "quantity": item.get("quantity", 1.0)
                        })
                    else:
                        print(f"Warning: Dish '{dish_name}' in set meal '{set_meal_name}' not found in catalog.")
                    
                set_meal = SetMeal(
                    name=set_meal_name,
                    included_dishes=valid_included_dishes,
                    set_meal_price=set_meal_info.get("set_meal_price", 0.0),
                    set_meal_discount=set_meal_info.get("set_meal_discount", 1.0)
                )
                self.set_meals[set_meal_name] = set_meal

        # Initialize user orders
        if "user_orders" in data:
            user_orders_data = data["user_orders"]
            self.user_orders.clear()
            # Iterate user order list
            for user_order_info in user_orders_data:
                user_id = user_order_info.get("user_id")
                if not user_id:
                    print("Warning: Order without user_id is skipped")
                    continue
                    
                self.user_orders[user_id] = {}
                # Iterate order item list for this user
                order_items = user_order_info.get("items", [])
                for item_info in order_items:
                    dish_name = item_info.get("dish_name", " ").lower()
                    # Ensure dish exists in catalog
                    if dish_name in self.catalog:
                        order_item = OrderItem(
                            dish_name=dish_name,
                            quantity=item_info.get("quantity", 1.0)
                        )
                        self.user_orders[user_id][dish_name] = order_item
                    # Check if it's a set meal
                    elif dish_name in self.set_meals:
                        order_item = OrderItem(
                            dish_name=dish_name,
                            quantity=item_info.get("quantity", 1.0)
                        )
                        self.user_orders[user_id][dish_name] = order_item
                    else:
                        print(f"Warning: Dish '{dish_name}' in user '{user_id}' order not found in catalog or set meals.")

    # --- Dish Catalog Management Tools ---

    def add_dish_to_catalog(self, name: str, category: str, price: float, tax_rate: float, discount: float,
                           nutritional_characteristics: List[str], taste: List[str], allergens: List[str],
                           nutrition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add or update a dish in the restaurant's catalog.
        """
        try:
            nutrition_obj = NutritionInfo(**nutrition)
            dish = Dish(
                name=name.lower(),
                category=category.lower(),
                price=price,
                tax_rate=tax_rate,
                discount=discount,
                nutritional_characteristics=[char.lower() for char in nutritional_characteristics],
                taste=[t.lower() for t in taste],
                allergens=[a.lower() for a in allergens],
                nutrition=nutrition_obj
            )
            self.catalog[name.lower()] = dish
            return {"status": "success", "message": f"Dish '{name}' added/updated successfully."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to add/update dish: {str(e)}"}

    def remove_dish_from_catalog(self, name: str) -> Dict[str, Any]:
        """
        Remove a dish from the catalog by its exact name.
        """
        dish_key = name.lower()
        if dish_key in self.catalog:
            del self.catalog[dish_key]
            # Also remove from any user's order where it might exist
            for order in self.user_orders.values():
                order.pop(dish_key, None)
            return {"status": "success", "message": f"Dish '{name}' removed from catalog."}
        else:
            return {"status": "error", "message": f"Dish '{name}' not found in catalog."}

    def update_dish_price(self, name: str, new_price: float) -> Dict[str, Any]:
        """
        Update the price of an existing dish in the catalog.
        """
        dish_key = name.lower()
        if dish_key in self.catalog:
            self.catalog[dish_key].price = new_price
            return {"status": "success", "message": f"Price of dish '{name}' updated to {new_price}."}
        else:
            return {"status": "error", "message": f"Dish '{name}' not found in catalog."}

    def update_dish_discount(self, name: str, new_discount: float) -> Dict[str, Any]:
        """
        Update the discount factor for an existing dish.
        """
        dish_key = name.lower()
        if dish_key in self.catalog:
            self.catalog[dish_key].discount = new_discount
            return {"status": "success", "message": f"Discount of dish '{name}' updated to {new_discount}."}
        else:
            return {"status": "error", "message": f"Dish '{name}' not found in catalog."}

    def find_dishes_by_category(self, category: str) -> Dict[str, Any]:
        """
        List all dishes belonging to a specific category.
        """
        cat_lower = category.lower()
        matching_dishes = [dish.name for dish in self.catalog.values() if dish.category == cat_lower]
        return {"dishes": matching_dishes}

    def list_all_dishes(self) -> Dict[str, Any]:
        """
        List all individual dish names currently available in the restaurant's full menu/catalog.
        """
        return {"dishes": [dish.name for dish in self.catalog.values()]}

    def find_dishes_by_nutritional_tag(self, tag: str) -> Dict[str, Any]:
        """
        Find all dishes that have a specific nutritional characteristic tag.
        """
        tag_lower = tag.lower()
        matching_dishes = [dish.name for dish in self.catalog.values() if tag_lower in dish.nutritional_characteristics]
        return {"dishes": matching_dishes}

    def find_dishes_by_allergen(self, allergen: str) -> Dict[str, Any]:
        """
        Find all dishes that contain a specific allergen.
        """
        allergen_lower = allergen.lower()
        matching_dishes = [dish.name for dish in self.catalog.values() if allergen_lower in dish.allergens]
        return {"dishes": matching_dishes}

    def find_dishes_by_taste(self, taste: str) -> Dict[str, Any]:
        """
        Find all dishes that match a specific taste profile.
        """
        taste_lower = taste.lower()
        matching_dishes = [dish.name for dish in self.catalog.values() if taste_lower in dish.taste]
        return {"dishes": matching_dishes}

    def filter_dishes_by_price_range(self, min_price: float, max_price: float) -> Dict[str, Any]:
        """
        Find all dishes with a price within a specified inclusive range.
        """
        matching_dishes = [
            dish.name for dish in self.catalog.values()
            if min_price <= dish.price <= max_price
        ]
        return {"dishes": matching_dishes}

    def list_all_discounted_dishes(self) -> Dict[str, Any]:
        """
        Return a list of all dish names that currently have a discount factor less than 1.0.
        """
        discounted_dishes = [name for name, dish in self.catalog.items() if dish.discount < 1.0]
        return {"discounted_dishes": discounted_dishes}

    # --- Dish Information Retrieval Tools (Use Fuzzy Search) ---

    def get_dish_nutrition(self, dish_name: str) -> Dict[str, Any]:
        """Retrieve detailed nutrition information for dishes by fuzzy name match."""
        matching_dishes = self._find_matching_dishes(dish_name)
        if matching_dishes:
            nutrition_data = {
                dish.name: asdict(dish.nutrition)
                for dish in matching_dishes
            }
            return {"status": "success", "matching_dishes": nutrition_data}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    def get_tax_rate(self, dish_name: str) -> Dict[str, Any]:
        """Get the tax rate for dishes by fuzzy name match."""
        matching_dishes = self._find_matching_dishes(dish_name)
        if matching_dishes:
            tax_rates = {dish.name: dish.tax_rate for dish in matching_dishes}
            return {"status": "success", "matching_dishes": tax_rates}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}
            
    def get_dish_allergens(self, dish_name: str) -> Dict[str, Any]:
        """Retrieve allergen information for dishes by fuzzy name match."""
        matching_dishes = self._find_matching_dishes(dish_name)
        if matching_dishes:
            allergens_data = {
                dish.name: dish.allergens
                for dish in matching_dishes
            }
            return {"status": "success", "matching_dishes": allergens_data}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    def get_dish_taste_profile(self, dish_name: str) -> Dict[str, Any]:
        """Retrieve the taste profile for dishes by fuzzy name match."""
        matching_dishes = self._find_matching_dishes(dish_name)
        if matching_dishes:
            taste_data = {
                dish.name: dish.taste
                for dish in matching_dishes
            }
            return {"status": "success", "matching_dishes": taste_data}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    def get_dish_price(self, dish_name: str) -> Dict[str, Any]:
        """Get the current shelf price for dishes by fuzzy name match."""
        matching_dishes = self._find_matching_dishes(dish_name)
        if matching_dishes:
            prices = {dish.name: dish.price for dish in matching_dishes}
            return {"status": "success", "matching_dishes": prices}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    def get_dish_discount(self, dish_name: str) -> Dict[str, Any]:
        """Get the current discount factor for dishes by fuzzy name match."""
        matching_dishes = self._find_matching_dishes(dish_name)
        if matching_dishes:
            discounts = {dish.name: dish.discount for dish in matching_dishes}
            return {"status": "success", "matching_dishes": discounts}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    # --- Set Meal Management Tools ---

    def create_set_meal(self, set_meal_name: str, included_dishes: List[Dict[str, Any]], 
                       set_meal_price: float, set_meal_discount: float) -> Dict[str, Any]:
        """
        Create a new set meal by combining existing dishes.
        """
        try:
            # Validate that all included dishes exist in the catalog
            for item in included_dishes:
                dish_name = item.get("dish_name", " ").lower()
                if dish_name not in self.catalog:
                    return {"status": "error", "message": f"Dish '{item['dish_name']}' not found in catalog."}
            
            set_meal = SetMeal(
                name=set_meal_name.lower(),
                included_dishes=included_dishes,
                set_meal_price=set_meal_price,
                set_meal_discount=set_meal_discount
            )
            self.set_meals[set_meal_name.lower()] = set_meal
            return {"status": "success", "message": f"Set meal '{set_meal_name}' created successfully."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to create set meal: {str(e)}"}

    def get_set_meal_details(self, set_meal_name: str) -> Dict[str, Any]:
        """Retrieve the composition, price, and discount of a set meal by its name."""
        set_meal_key = set_meal_name.lower()
        if set_meal_key in self.set_meals:
            set_meal = self.set_meals[set_meal_key]
            if set_meal.set_meal_price and set_meal.set_meal_price > 0:
                price = set_meal.set_meal_price
            else:
                price = 0.0
                for included_item in set_meal.included_dishes:
                    included_dish_name = included_item.get("dish_name", " ").lower()
                    included_qty = included_item.get("quantity", 1.0)
                    if included_dish_name in self.catalog:
                        included_dish = self.catalog[included_dish_name]
                        price += included_dish.price * included_dish.discount * included_qty
            return {
                "name": set_meal.name,
                "included_dishes": set_meal.included_dishes,
                "price": price,
                "discount": set_meal.set_meal_discount
            }
        else:
            return {"status": "error", "message": f"Set meal '{set_meal_name}' not found."}

    def find_set_meals_containing_dish(self, dish_name: str) -> Dict[str, Any]:
        """Find all set meals that include a specific dish by name."""
        dish_key = dish_name.lower()
        matching_set_meals = []
        
        for set_meal in self.set_meals.values():
            for item in set_meal.included_dishes:
                if item.get("dish_name", " ").lower() == dish_key:
                    matching_set_meals.append(set_meal.name)
                    break  # Break inner loop once we find the dish in this set meal
        
        return {"set_meals": matching_set_meals}

    # --- Order Management Tools ---

    def add_dish_to_order(self, user_id: str, dish_name: str, quantity: float) -> Dict[str, Any]:
        """Add one or more servings of a dish to a user's current order."""
        dish_key = dish_name.lower()
        if dish_key not in self.catalog:
            return {"status": "error", "message": f"Dish '{dish_name}' does not exist in catalog."}

        if user_id not in self.user_orders:
            self.user_orders[user_id] = {}

        if dish_key in self.user_orders[user_id]:
            self.user_orders[user_id][dish_key].quantity += quantity
        else:
            self.user_orders[user_id][dish_key] = OrderItem(dish_name=dish_key, quantity=quantity)

        return {"status": "success", "message": f"Added {quantity} x {dish_name} to order for user {user_id}."}

    def remove_dish_from_order(self, user_id: str, dish_name: str, quantity: float) -> Dict[str, Any]:
        """Remove a specified quantity of a dish from a user's current order."""
        dish_key = dish_name.lower()
        if user_id not in self.user_orders or dish_key not in self.user_orders[user_id]:
            return {"status": "error", "message": f"Dish '{dish_name}' not found in order for user {user_id}."}

        current_qty = self.user_orders[user_id][dish_key].quantity
        new_qty = current_qty - quantity

        if new_qty <= 0:
            del self.user_orders[user_id][dish_key]
        else:
            self.user_orders[user_id][dish_key].quantity = new_qty

        action_msg = "Removed " if new_qty >= 0 else "Removed entire item (requested more than available)"
        return {"status": "success", "message": f"{action_msg} from order for user {user_id}."}

    def clear_user_order(self, user_id: str) -> Dict[str, Any]:
        """Clear all items from a user's current order."""
        if user_id in self.user_orders:
            self.user_orders[user_id] = {}
        return {"status": "success", "message": f"Order cleared for user {user_id}."}

    def get_user_order_summary(self, user_id: str) -> Dict[str, Any]:
        """Retrieve all dishes and set meals with quantities in a user's current order."""
        if user_id not in self.user_orders:
            return {"user_id": user_id, "items": [], "total_items": 0}

        items = []
        for item in self.user_orders[user_id].values():
            item_key = item.dish_name.lower()
            item_type = "set_meal" if item_key in self.set_meals else "dish"
            items.append({
                "dish_name": item.dish_name,
                "item_type": item_type,
                "quantity": item.quantity,
            })
        return {"user_id": user_id, "items": items, "total_items": len(items)}

    # def calculate_order_total(self, user_id: str) -> Dict[str, Any]:
    #     """Calculate the total payable amount for a user's current order: sum(price * discount * quantity)."""
    #     if user_id not in self.user_orders:
    #         return {"user_id": user_id, "total": 0.0}

    #     total = 0.0
    #     for item in self.user_orders[user_id].values():
    #         dish = self.catalog.get(item.dish_name)
    #         if dish:
    #             item_total = dish.price * dish.discount * item.quantity
    #             total += item_total

    #     return {"user_id": user_id, "total": round(total, 2)}

    # def calculate_order_tax(self, user_id: str) -> Dict[str, Any]:
    #     """
    #     Calculate the total tax amount included in a user's current order.
    #     Tax per item = (price * tax_rate / (1 + tax_rate)) * discount * quantity.
    #     """
    #     if user_id not in self.user_orders:
    #         return {"user_id": user_id, "total_tax": 0.0}

    #     total_tax = 0.0
    #     for item in self.user_orders[user_id].values():
    #         dish = self.catalog.get(item.dish_name)
    #         if dish:
    #             pre_tax_price_per_unit = (dish.price * dish.tax_rate) / (1 + dish.tax_rate)
    #             discounted_pre_tax_price_per_unit = pre_tax_price_per_unit * dish.discount
    #             item_tax = discounted_pre_tax_price_per_unit * item.quantity
    #             total_tax += item_tax

    #     return {"user_id": user_id, "total_tax": round(total_tax, 2)}

    # def summarize_order_nutrition(self, user_id: str) -> Dict[str, Any]:
    #     """Calculate the total aggregated nutrition values across all items in a user's current order."""
    #     if user_id not in self.user_orders:
    #         return {"user_id": user_id, "total_nutrition": {}}

    #     total_nutrition = {
    #         "calories_kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0,
    #         "carbs_g": 0.0, "sugar_g": 0.0, "sodium_mg": 0.0, "fiber_g": 0.0
    #     }

    #     for item in self.user_orders[user_id].values():
    #         dish = self.catalog.get(item.dish_name)
    #         if dish and dish.nutrition:
    #             nut_info = dish.nutrition
    #             multiplier = item.quantity / 100.0 if nut_info.basis == "PER_100G" else item.quantity
    #             total_nutrition["calories_kcal"] += (nut_info.calories_kcal or 0) * multiplier
    #             total_nutrition["protein_g"] += (nut_info.protein_g or 0) * multiplier
    #             total_nutrition["fat_g"] += (nut_info.fat_g or 0) * multiplier
    #             total_nutrition["carbs_g"] += (nut_info.carbs_g or 0) * multiplier
    #             total_nutrition["sugar_g"] += (nut_info.sugar_g or 0) * multiplier
    #             total_nutrition["sodium_mg"] += (nut_info.sodium_mg or 0) * multiplier
    #             total_nutrition["fiber_g"] += (nut_info.fiber_g or 0) * multiplier

    #     rounded_nutrition = {k: round(v, 2) for k, v in total_nutrition.items()}
    #     return {"user_id": user_id, "total_nutrition": rounded_nutrition}

    def compute_total_payment(self, user_id: str, dishes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute total payable amount for the specified dishes: sum(price * discount * qty).
        """
        if not dishes:
            return {"user_id": user_id, "total": 0.0}

        total = 0.0
        for item in dishes:
            dish_name = item.get("dish_name", " ").lower()
            quantity = item.get("quantity", 0)

            if dish_name in self.catalog:
                dish = self.catalog[dish_name]
                item_total = dish.price * dish.discount * quantity
                total += item_total
            elif dish_name in self.set_meals:
                # Check if it's a set meal
                set_meal = self.set_meals[dish_name]
                if set_meal.set_meal_price and set_meal.set_meal_price > 0:
                    item_total = set_meal.set_meal_price * set_meal.set_meal_discount * quantity
                else:
                    meal_payment = 0.0
                    for included_item in set_meal.included_dishes:
                        included_dish_name = included_item.get("dish_name", " ").lower()
                        included_qty = included_item.get("quantity", 1.0)
                        if included_dish_name in self.catalog:
                            included_dish = self.catalog[included_dish_name]
                            meal_payment += included_dish.price * included_dish.discount * included_qty
                    item_total = meal_payment * set_meal.set_meal_discount * quantity
                total += item_total
            else:
                # Optionally log warning for missing dish
                pass

        return {"user_id": user_id, "total": round(total, 2)}

    def compute_total_tax(self, user_id: str, dishes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute total tax amount for the specified dishes.
        Since price is tax-inclusive, tax per item = (price * tax_rate / (1 + tax_rate)) * discount * qty.
        """
        if not dishes:
            return {"user_id": user_id, "total_tax": 0.0}

        total_tax = 0.0
        for item in dishes:
            dish_name = item.get("dish_name", " ").lower()
            quantity = item.get("quantity", 0)

            if dish_name in self.catalog:
                dish = self.catalog[dish_name]
                pre_tax_price_per_unit = (dish.price * dish.tax_rate) / (1 + dish.tax_rate)
                discounted_pre_tax_price_per_unit = pre_tax_price_per_unit * dish.discount
                item_tax = discounted_pre_tax_price_per_unit * quantity
                total_tax += item_tax
            elif dish_name in self.set_meals:
                # Check if it's a set meal - calculate tax based on included dishes
                set_meal = self.set_meals[dish_name]
                for included_item in set_meal.included_dishes:
                    included_dish_name = included_item.get("dish_name", " ").lower()
                    included_qty = included_item.get("quantity", 1.0)
                    if included_dish_name in self.catalog:
                        included_dish = self.catalog[included_dish_name]
                        pre_tax_price = (included_dish.price * included_dish.tax_rate) / (1 + included_dish.tax_rate)
                        discounted_pre_tax_price = pre_tax_price * included_dish.discount
                        item_tax = discounted_pre_tax_price * included_qty * quantity
                        total_tax += item_tax
            else:
                pass

        return {"user_id": user_id, "total_tax": round(total_tax, 2)}

    def compute_total_nutrition(self, user_id: str, dishes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute total nutrition values across all items for the specified dishes.
        """
        if not dishes:
            return {"user_id": user_id, "total_nutrition": {}}

        total_nutrition = {
            "calories_kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0,
            "carbs_g": 0.0, "sugar_g": 0.0, "sodium_mg": 0.0, "fiber_g": 0.0
        }

        for item in dishes:
            dish_name = item.get("dish_name", " ").lower()
            quantity = item.get("quantity", 0)

            if dish_name in self.catalog:
                dish = self.catalog[dish_name]
                if dish and dish.nutrition:
                    nut_info = dish.nutrition
                    # Determine multiplier based on nutrition basis
                    multiplier = quantity / 100.0 if nut_info.basis == "PER_100G" else quantity

                    total_nutrition["calories_kcal"] += (nut_info.calories_kcal or 0) * multiplier
                    total_nutrition["protein_g"] += (nut_info.protein_g or 0) * multiplier
                    total_nutrition["fat_g"] += (nut_info.fat_g or 0) * multiplier
                    total_nutrition["carbs_g"] += (nut_info.carbs_g or 0) * multiplier
                    total_nutrition["sugar_g"] += (nut_info.sugar_g or 0) * multiplier
                    total_nutrition["sodium_mg"] += (nut_info.sodium_mg or 0) * multiplier
                    total_nutrition["fiber_g"] += (nut_info.fiber_g or 0) * multiplier
            elif dish_name in self.set_meals:
                # Check if it's a set meal - calculate nutrition based on included dishes
                set_meal = self.set_meals[dish_name]
                for included_item in set_meal.included_dishes:
                    included_dish_name = included_item.get("dish_name", " ").lower()
                    included_qty = included_item.get("quantity", 1.0)
                    if included_dish_name in self.catalog:
                        included_dish = self.catalog[included_dish_name]
                        if included_dish and included_dish.nutrition:
                            nut_info = included_dish.nutrition
                            multiplier = (included_qty * quantity) / 100.0 if nut_info.basis == "PER_100G" else included_qty * quantity

                            total_nutrition["calories_kcal"] += (nut_info.calories_kcal or 0) * multiplier
                            total_nutrition["protein_g"] += (nut_info.protein_g or 0) * multiplier
                            total_nutrition["fat_g"] += (nut_info.fat_g or 0) * multiplier
                            total_nutrition["carbs_g"] += (nut_info.carbs_g or 0) * multiplier
                            total_nutrition["sugar_g"] += (nut_info.sugar_g or 0) * multiplier
                            total_nutrition["sodium_mg"] += (nut_info.sodium_mg or 0) * multiplier
                            total_nutrition["fiber_g"] += (nut_info.fiber_g or 0) * multiplier

        rounded_nutrition = {k: round(v, 2) for k, v in total_nutrition.items()}
        return {"user_id": user_id, "total_nutrition": rounded_nutrition}


    # --- Set Meal Order Management ---

    def add_set_meal_to_order(self, user_id: str, set_meal_name: str, quantity: float) -> Dict[str, Any]:
        """Add a pre-defined set meal to a user's order."""
        set_meal_key = set_meal_name.lower()
        if set_meal_key not in self.set_meals:
            return {"status": "error", "message": f"Set meal '{set_meal_name}' does not exist."}

        if user_id not in self.user_orders:
            self.user_orders[user_id] = {}

        # For simplicity, treat set meal as a special dish in the order
        # In a real system, we might want to track it differently
        if set_meal_key in self.user_orders[user_id]:
            self.user_orders[user_id][set_meal_key].quantity += quantity
        else:
            self.user_orders[user_id][set_meal_key] = OrderItem(dish_name=set_meal_key, quantity=quantity)

        return {"status": "success", "message": f"Added {quantity} x {set_meal_name} set meal to order for user {user_id}."}

    def remove_set_meal_from_order(self, user_id: str, set_meal_name: str, quantity: float) -> Dict[str, Any]:
        """Remove a specified quantity of a set meal from a user's order."""
        set_meal_key = set_meal_name.lower()
        order_key = set_meal_key

        if user_id not in self.user_orders or order_key not in self.user_orders[user_id]:
            return {"status": "error", "message": f"Set meal '{set_meal_name}' not found in order for user {user_id}."}

        current_qty = self.user_orders[user_id][order_key].quantity
        new_qty = current_qty - quantity

        if new_qty <= 0:
            del self.user_orders[user_id][order_key]
        else:
            self.user_orders[user_id][order_key].quantity = new_qty

        return {"status": "success", "message": f"Removed {quantity} x {set_meal_name} set meal from order for user {user_id}."}

import difflib
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
import json

# ======================
# Data Models
# ======================
@dataclass
class NutritionInfo:
    basis: str
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
    name: str
    category: str
    price: float
    tax_rate: float
    discount: float
    nutritional_characteristics: List[str]
    taste: List[str]
    allergens: List[str]
    nutrition: NutritionInfo
    restaurant_name: str

@dataclass
class SetMeal:
    name: str
    included_dishes: List[Dict[str, Any]]
    set_meal_price: float
    set_meal_discount: float
    restaurant_name: str

@dataclass
class OrderItem:
    dish_name: str
    quantity: float

# ======================
# Order Database Class
# ======================
class OrderDB:
    """
    Unified Order Database Management System:
    - Manage dish catalogs, set meals, and user orders for multiple restaurants
    - Restaurant related operations: must provide restaurant_name
    - user_current_restaurant: Save the current last active restaurant name (string, unrelated to user_id)
    """
    def __init__(self):
        # Restaurant name -> Data storage dictionary {'catalog': {}, 'set_meals': {}, 'user_orders': {}}
        self.restaurants: Dict[str, Dict[str, Any]] = {}
        # Current last active restaurant name (string, unrelated to user_id)
        self.user_current_restaurant: Optional[str] = None

    # --- Internal Helper Methods ---
    def _get_store(self, restaurant_name: str) -> Dict[str, Any]:
        """Get or create restaurant data storage dictionary"""
        if restaurant_name not in self.restaurants:
            self.restaurants[restaurant_name] = {
                'catalog': {},
                'set_meals': {},
                'user_orders': {}
            }
        return self.restaurants[restaurant_name]

    def _resolve_store(self, restaurant_name: str) -> Optional[Dict[str, Any]]:
        """Resolve restaurant store by exact or case-insensitive name."""
        if restaurant_name in self.restaurants:
            return self.restaurants[restaurant_name]
        restaurant_lower = (restaurant_name or "").lower()
        for stored_name, store in self.restaurants.items():
            if stored_name.lower() == restaurant_lower:
                return store
        return None

    def _find_matching_dishes(self, restaurant_name: str, query: str) -> List[Dish]:
        store = self._get_store(restaurant_name)
        query_lower = query.lower().strip()
        if query_lower in store['catalog']:
            return [store['catalog'][query_lower]]

        matching_dishes = []

        for dish in store['catalog'].values():
            dish_name_lower = dish.name.lower()
            if query_lower in dish_name_lower or (len(dish_name_lower) > 1 and dish_name_lower in query_lower):
                matching_dishes.append(dish)

        return matching_dishes

    def _resolve_order_item_key(self, restaurant_name: str, order: Dict[str, OrderItem], dish_name: str) -> Optional[str]:
        """Resolve an order item key by exact key or dish catalog fuzzy match."""
        dish_key = dish_name.lower()
        if dish_key in order:
            return dish_key
        wanted = {dish.name for dish in self._find_matching_dishes(restaurant_name, dish_name)}
        for key in order:
            if key in wanted:
                return key
        return None

    def _find_matching_set_meals(self, restaurant_name: str, query: str) -> List[SetMeal]:
        store = self._get_store(restaurant_name)
        query_lower = query.lower()
        matching_set_meals = []

        for set_meal in store['set_meals'].values():
            set_meal_name_lower = set_meal.name.lower()
            if query_lower in set_meal_name_lower or set_meal_name_lower in query_lower:
                matching_set_meals.append(set_meal)

        return matching_set_meals

    # ======================
    # Initialization Method (Adapt to Flat JSON Format)
    # ======================
    def init_from_json(self, data: Dict[str, Any]) -> None:
        """
        Initialize the system from a flat JSON structure.

        Expected JSON structure:
        {
            "dishes": [{ ..., "restaurant_name": "..." }, ... ],
            "set_meals": [{ ..., "restaurant_name": "..." }, ... ],
            "user_orders": [{ "user_id": "...", "items": [...], "restaurant_name": "..." }, ... ]
        }
        """
        # Clear existing data
        self.restaurants.clear()
        self.user_current_restaurant = None

        # 1. Initialize dishes (must be before set meals and orders to verify existence)
        if "dishes" in data:
            for dish_info in data["dishes"]:
                restaurant_name = dish_info.get("restaurant_name")
                if not restaurant_name:
                    print("Warning: Dish without 'restaurant_name' skipped.")
                    continue

                store = self._get_store(restaurant_name)
                dish_name = dish_info.get("name", " ").lower()
                if not dish_name:
                    print("Warning: Dish without name is skipped")
                    continue

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
                    nutrition=nutrition_obj,
                    restaurant_name=restaurant_name
                )
                store['catalog'][dish_name] = dish

        # 2. Initialize set meals
        if "set_meals" in data:
            for set_meal_info in data["set_meals"]:
                restaurant_name = set_meal_info.get("restaurant_name")
                if not restaurant_name:
                    print("Warning: Set meal without 'restaurant_name' skipped.")
                    continue

                store = self._get_store(restaurant_name)
                set_meal_name = set_meal_info.get("name", " ").lower()
                if not set_meal_name:
                    print("Warning: Set meal without name is skipped")
                    continue

                included_dishes = set_meal_info.get("included_dishes", [])
                valid_included_dishes = []
                for item in included_dishes:
                    dish_name = item.get("dish_name", " ").lower()
                    if dish_name in store['catalog']:
                        valid_included_dishes.append({
                            "dish_name": dish_name,
                            "quantity": item.get("quantity", 1.0)
                        })
                    else:
                        print(f"Warning: Dish '{dish_name}' in set meal '{set_meal_name}' not found in catalog for '{restaurant_name}'.")

                set_meal = SetMeal(
                    name=set_meal_name,
                    included_dishes=valid_included_dishes,
                    set_meal_price=set_meal_info.get("set_meal_price", 0.0),
                    set_meal_discount=set_meal_info.get("set_meal_discount", 1.0),
                    restaurant_name=restaurant_name
                )
                store['set_meals'][set_meal_name] = set_meal

        # 3. Initialize user orders
        if "user_orders" in data:
            for user_order_info in data["user_orders"]:
                user_id = user_order_info.get("user_id")
                if not user_id:
                    print("Warning: Order without user_id is skipped")
                    continue

                restaurant_name = user_order_info.get("restaurant_name")

                # If restaurant not specified, try to infer from order dishes
                if not restaurant_name:
                    items = user_order_info.get("items", [])
                    for item in items:
                        dish_name = item.get("dish_name", " ").lower()
                        for r_name, store in self.restaurants.items():
                            if dish_name in store['catalog']:
                                restaurant_name = r_name
                                break
                        if restaurant_name:
                            break

                    if not restaurant_name:
                        if self.restaurants:
                            restaurant_name = list(self.restaurants.keys())[0]
                            print(f"Warning: Restaurant for user '{user_id}' order inferred as '{restaurant_name}' (default).")
                        else:
                            print(f"Warning: Order for user '{user_id}' skipped (no restaurant context).")
                            continue

                store = self._get_store(restaurant_name)

                if user_id not in store['user_orders']:
                    store['user_orders'][user_id] = {}

                order_items = user_order_info.get("items", [])
                for item_info in order_items:
                    dish_name = item_info.get("dish_name", " ").lower()
                    # Check if it's a dish in catalog
                    if dish_name in store['catalog']:
                        order_item = OrderItem(
                            dish_name=dish_name,
                            quantity=item_info.get("quantity", 1.0)
                        )
                        store['user_orders'][user_id][dish_name] = order_item
                    # Check if it's a set meal
                    elif dish_name in store['set_meals']:
                        order_item = OrderItem(
                            dish_name=dish_name,
                            quantity=item_info.get("quantity", 1.0)
                        )
                        store['user_orders'][user_id][dish_name] = order_item
                    else:
                        # Fallback: add as-is (backward compatibility)
                        order_item = OrderItem(
                            dish_name=dish_name,
                            quantity=item_info.get("quantity", 1.0)
                        )
                        store['user_orders'][user_id][dish_name] = order_item

                # Update current active restaurant (string, unrelated to user_id)
                self.user_current_restaurant = restaurant_name

        print(f"✓ Initialization complete. Total restaurants: {len(self.restaurants)}")
        if self.user_current_restaurant:
            print(f"✓ Current active restaurant: {self.user_current_restaurant}")

    # ======================
    # Catalog Management Tools
    # ======================
    def add_dish_to_catalog(self, restaurant_name: str, category: str, price: float,
                            tax_rate: float, discount: float, nutritional_characteristics: List[str],
                            taste: List[str], allergens: List[str], nutrition: Dict[str, Any],
                            name: str = None, dish_name: str = None) -> Dict[str, Any]:
        """Add dish to specified restaurant's catalog"""
        store = self._get_store(restaurant_name)
        name = name or dish_name
        if not name:
            return {"status": "error", "message": "Dish name is required."}
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
                nutrition=nutrition_obj,
                restaurant_name=restaurant_name
            )
            store['catalog'][name.lower()] = dish
            # Update current active restaurant
            self.user_current_restaurant = restaurant_name
            return {"status": "success", "message": f"Dish '{name}' added/updated successfully."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to add/update dish: {str(e)}"}

    def remove_dish_from_catalog(self, restaurant_name: str, name: str = None, dish_name: str = None) -> Dict[str, Any]:
        """Remove dish from specified restaurant's catalog"""
        store = self._get_store(restaurant_name)
        name = name or dish_name
        if not name:
            return {"status": "error", "message": "Dish name is required."}
        dish_key = name.lower()
        if dish_key in store['catalog']:
            del store['catalog'][dish_key]
            for order in store['user_orders'].values():
                order.pop(dish_key, None)
            return {"status": "success", "message": f"Dish '{name}' removed from catalog."}
        else:
            return {"status": "error", "message": f"Dish '{name}' not found in catalog."}

    def update_dish_price(self, restaurant_name: str, new_price: float, name: str = None, dish_name: str = None) -> Dict[str, Any]:
        """Update price of dish in specified restaurant"""
        store = self._get_store(restaurant_name)
        name = name or dish_name
        if not name:
            return {"status": "error", "message": "Dish name is required."}
        dish_key = name.lower()
        if dish_key in store['catalog']:
            store['catalog'][dish_key].price = new_price
            # Update current active restaurant
            self.user_current_restaurant = restaurant_name
            return {"status": "success", "message": f"Price of dish '{name}' updated to {new_price}."}
        else:
            return {"status": "error", "message": f"Dish '{name}' not found in catalog."}

    def update_dish_discount(self, restaurant_name: str, new_discount: float, name: str = None, dish_name: str = None) -> Dict[str, Any]:
        """Update discount of dish in specified restaurant"""
        store = self._get_store(restaurant_name)
        name = name or dish_name
        if not name:
            return {"status": "error", "message": "Dish name is required."}
        dish_key = name.lower()
        if dish_key in store['catalog']:
            store['catalog'][dish_key].discount = new_discount
            # Update current active restaurant
            self.user_current_restaurant = restaurant_name
            return {"status": "success", "message": f"Discount of dish '{name}' updated to {new_discount}."}
        else:
            return {"status": "error", "message": f"Dish '{name}' not found in catalog."}

    def find_dishes_by_category(self, restaurant_name: str, category: str) -> Dict[str, Any]:
        """Find dishes by category in specified restaurant"""
        store = self._get_store(restaurant_name)
        cat_lower = category.lower()
        matching_dishes = [dish.name for dish in store['catalog'].values() if dish.category == cat_lower]
        return {"dishes": matching_dishes}

    def find_dishes_by_nutritional_tag(self, restaurant_name: str, tag: str) -> Dict[str, Any]:
        """Find dishes by nutritional tag in specified restaurant"""
        store = self._get_store(restaurant_name)
        tag_lower = tag.lower()
        matching_dishes = [dish.name for dish in store['catalog'].values() if tag_lower in dish.nutritional_characteristics]
        return {"dishes": matching_dishes}

    def find_dishes_by_taste(self, restaurant_name: str, taste: str) -> Dict[str, Any]:
        """Find dishes by taste in specified restaurant"""
        store = self._get_store(restaurant_name)
        taste_lower = taste.lower()
        matching_dishes = [dish.name for dish in store['catalog'].values() if taste_lower in dish.taste]
        return {"dishes": matching_dishes}

    def find_dishes_by_allergen(self, restaurant_name: str, allergen: str) -> Dict[str, Any]:
        """Find dishes by allergen in specified restaurant"""
        store = self._get_store(restaurant_name)
        allergen_lower = allergen.lower()
        matching_dishes = [dish.name for dish in store['catalog'].values() if allergen_lower in dish.allergens]
        return {"dishes": matching_dishes}

    def filter_dishes_by_price_range(self, restaurant_name: str, min_price: float, max_price: float) -> Dict[str, Any]:
        """Filter dishes by price range in specified restaurant"""
        store = self._get_store(restaurant_name)
        matching_dishes = [
            dish.name for dish in store['catalog'].values()
            if min_price <= dish.price <= max_price
        ]
        return {"dishes": matching_dishes}

    def list_all_discounted_dishes(self, restaurant_name: str) -> Dict[str, Any]:
        """List all discounted dishes in specified restaurant"""
        store = self._get_store(restaurant_name)
        discounted_dishes = [name for name, dish in store['catalog'].items() if dish.discount < 1.0]
        return {"discounted_dishes": discounted_dishes}

    # ======================
    # Dish Information Query Tools
    # ======================
    def get_dish_nutrition(self, restaurant_name: str, dish_name: str) -> Dict[str, Any]:
        """Get nutrition information of dish in specified restaurant"""
        matching_dishes = self._find_matching_dishes(restaurant_name, dish_name)
        if matching_dishes:
            nutrition_data = {
                dish.name: asdict(dish.nutrition)
                for dish in matching_dishes
            }
            return {"status": "success", "matching_dishes": nutrition_data}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    def get_tax_rate(self, restaurant_name: str, dish_name: str) -> Dict[str, Any]:
        """Get nutrition information of dish in specified restaurant"""
        matching_dishes = self._find_matching_dishes(restaurant_name, dish_name)
        if matching_dishes:
            tax_data = {
                dish.name: dish.tax_rate
                for dish in matching_dishes
            }
            return {"status": "success", "matching_dishes": tax_data}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    def get_dish_allergens(self, restaurant_name: str, dish_name: str) -> Dict[str, Any]:
        """Get allergen information of dish in specified restaurant"""
        matching_dishes = self._find_matching_dishes(restaurant_name, dish_name)
        if matching_dishes:
            allergens_data = {
                dish.name: dish.allergens
                for dish in matching_dishes
            }
            return {"status": "success", "matching_dishes": allergens_data}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    def get_dish_taste_profile(self, restaurant_name: str, dish_name: str) -> Dict[str, Any]:
        """Get taste information of dish in specified restaurant"""
        matching_dishes = self._find_matching_dishes(restaurant_name, dish_name)
        if matching_dishes:
            taste_data = {
                dish.name: dish.taste
                for dish in matching_dishes
            }
            return {"status": "success", "matching_dishes": taste_data}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    def get_dish_price(self, restaurant_name: str, dish_name: str) -> Dict[str, Any]:
        """Get price of dish in specified restaurant"""
        matching_dishes = self._find_matching_dishes(restaurant_name, dish_name)
        if matching_dishes:
            prices = {dish.name: dish.price for dish in matching_dishes}
            return {"status": "success", "matching_dishes": prices}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    def get_dish_discount(self, restaurant_name: str, dish_name: str) -> Dict[str, Any]:
        """Get discount of dish in specified restaurant"""
        matching_dishes = self._find_matching_dishes(restaurant_name, dish_name)
        if matching_dishes:
            discounts = {dish.name: dish.discount for dish in matching_dishes}
            return {"status": "success", "matching_dishes": discounts}
        else:
            return {"status": "error", "message": f"No dishes found containing '{dish_name}'."}

    # ======================
    # Set Meal Management Tools
    # ======================
    def create_set_meal(self, restaurant_name: str, set_meal_name: str, included_dishes: List[Dict[str, Any]],
                        set_meal_price: float, set_meal_discount: float) -> Dict[str, Any]:
        """Create set meal in specified restaurant"""
        store = self._get_store(restaurant_name)
        try:
            for item in included_dishes:
                dish_name = item.get("dish_name", " ").lower()
                if dish_name not in store['catalog']:
                    return {"status": "error", "message": f"Dish '{item['dish_name']}' not found in catalog."}

            set_meal = SetMeal(
                name=set_meal_name.lower(),
                included_dishes=included_dishes,
                set_meal_price=set_meal_price,
                set_meal_discount=set_meal_discount,
                restaurant_name=restaurant_name
            )
            store['set_meals'][set_meal_name.lower()] = set_meal
            # Update current active restaurant
            self.user_current_restaurant = restaurant_name
            return {"status": "success", "message": f"Set meal '{set_meal_name}' created successfully."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to create set meal: {str(e)}"}

    def get_set_meal_details(self, restaurant_name: str, set_meal_name: str) -> Dict[str, Any]:
        """Get detailed information of set meal in specified restaurant"""
        store = self._get_store(restaurant_name)
        set_meal_key = set_meal_name.lower()
        if set_meal_key in store['set_meals']:
            set_meal = store['set_meals'][set_meal_key]
            if set_meal.set_meal_price and set_meal.set_meal_price > 0:
                price = set_meal.set_meal_price
            else:
                price = 0.0
                for included_item in set_meal.included_dishes:
                    included_dish_name = included_item.get("dish_name", " ").lower()
                    included_qty = included_item.get("quantity", 1.0)
                    if included_dish_name in store["catalog"]:
                        included_dish = store["catalog"][included_dish_name]
                        price += included_dish.price * included_dish.discount * included_qty
            return {
                "name": set_meal.name,
                "included_dishes": set_meal.included_dishes,
                "price": price,
                "discount": set_meal.set_meal_discount
            }
        else:
            return {"status": "error", "message": f"Set meal '{set_meal_name}' not found."}

    def find_set_meals_containing_dish(self, restaurant_name: str, dish_name: str) -> Dict[str, Any]:
        """Find set meals containing a specific dish in specified restaurant"""
        store = self._get_store(restaurant_name)
        dish_key = dish_name.lower()
        matching_set_meals = []

        for set_meal in store['set_meals'].values():
            for item in set_meal.included_dishes:
                if item.get("dish_name", " ").lower() == dish_key:
                    matching_set_meals.append(set_meal.name)
                    break

        return {"set_meals": matching_set_meals}

    # ======================
    # Order Management Tools
    # ======================
    def add_dish_to_order(self, restaurant_name: str, user_id: str, dish_name: str, quantity: float) -> Dict[str, Any]:
        """Add dish to user order in specified restaurant"""
        store = self._get_store(restaurant_name)
        dish_key = dish_name.lower()
        if dish_key not in store['catalog']:
            return {"status": "error", "message": f"Dish '{dish_name}' does not exist in catalog."}

        if user_id not in store['user_orders']:
            store['user_orders'][user_id] = {}

        if dish_key in store['user_orders'][user_id]:
            store['user_orders'][user_id][dish_key].quantity += quantity
        else:
            store['user_orders'][user_id][dish_key] = OrderItem(dish_name=dish_key, quantity=quantity)

        # Update current active restaurant (string, unrelated to user_id)
        self.user_current_restaurant = restaurant_name
        return {"status": "success", "message": f"Added {quantity} x {dish_name} to order for user {user_id}."}

    def remove_dish_from_order(self, restaurant_name: str, user_id: str, dish_name: str, quantity: float) -> Dict[str, Any]:
        """Remove dish from user order in specified restaurant"""
        store = self._get_store(restaurant_name)
        if user_id not in store['user_orders']:
            return {"status": "error", "message": f"Dish '{dish_name}' not found in order for user {user_id}."}

        dish_key = self._resolve_order_item_key(restaurant_name, store['user_orders'][user_id], dish_name)
        if dish_key is None:
            return {"status": "error", "message": f"Dish '{dish_name}' not found in order for user {user_id}."}

        current_qty = store['user_orders'][user_id][dish_key].quantity
        new_qty = current_qty - quantity

        if new_qty <= 0:
            del store['user_orders'][user_id][dish_key]
        else:
            store['user_orders'][user_id][dish_key].quantity = new_qty

        # Update current active restaurant
        self.user_current_restaurant = restaurant_name

        action_msg = "Removed " if new_qty >= 0 else "Removed entire item (requested more than available)"
        return {"status": "success", "message": f"{action_msg} from order for user {user_id}."}

    def clear_user_order(self, restaurant_name: str, user_id: str) -> Dict[str, Any]:
        """Clear user order in specified restaurant"""
        store = self._get_store(restaurant_name)
        if user_id in store['user_orders']:
            store['user_orders'][user_id] = {}
        # Clear active restaurant record
        self.user_current_restaurant = None
        return {"status": "success", "message": f"Order cleared for user {user_id}."}

    def get_user_order_summary(self, restaurant_name: str, user_id: str) -> Dict[str, Any]:
        """Get all dishes and set meals with quantities in a user's current restaurant order."""
        store = self._get_store(restaurant_name)
        if user_id not in store['user_orders']:
            return {"user_id": user_id, "items": [], "total_items": 0}

        items = []
        for item in store['user_orders'][user_id].values():
            item_key = item.dish_name.lower()
            item_type = "set_meal" if item_key in store['set_meals'] else "dish"
            items.append({
                "dish_name": item.dish_name,
                "item_type": item_type,
                "quantity": item.quantity,
            })
        return {"user_id": user_id, "items": items, "total_items": len(items)}

    # def calculate_order_total(self, restaurant_name: str, user_id: str) -> Dict[str, Any]:
    #     """Calculate total order price for user in specified restaurant"""
    #     store = self._get_store(restaurant_name)
    #     if user_id not in store['user_orders']:
    #         return {"user_id": user_id, "total": 0.0}

    #     total = 0.0
    #     for item in store['user_orders'][user_id].values():
    #         dish = store['catalog'].get(item.dish_name)
    #         if dish:
    #             item_total = dish.price * dish.discount * item.quantity
    #             total += item_total

    #     return {"user_id": user_id, "total": round(total, 2)}

    # def calculate_order_tax(self, restaurant_name: str, user_id: str) -> Dict[str, Any]:
    #     """Calculate total order tax for user in specified restaurant"""
    #     store = self._get_store(restaurant_name)
    #     if user_id not in store['user_orders']:
    #         return {"user_id": user_id, "total_tax": 0.0}

    #     total_tax = 0.0
    #     for item in store['user_orders'][user_id].values():
    #         dish = store['catalog'].get(item.dish_name)
    #         if dish:
    #             pre_tax_price_per_unit = (dish.price * dish.tax_rate) / (1 + dish.tax_rate)
    #             discounted_pre_tax_price_per_unit = pre_tax_price_per_unit * dish.discount
    #             item_tax = discounted_pre_tax_price_per_unit * item.quantity
    #             total_tax += item_tax

    #     return {"user_id": user_id, "total_tax": round(total_tax, 2)}

    # def summarize_order_nutrition(self, restaurant_name: str, user_id: str) -> Dict[str, Any]:
    #     """Calculate total order nutrition for user in specified restaurant"""
    #     store = self._get_store(restaurant_name)
    #     if user_id not in store['user_orders']:
    #         return {"user_id": user_id, "total_nutrition": {}}

    #     total_nutrition = {
    #         "calories_kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0,
    #         "carbs_g": 0.0, "sugar_g": 0.0, "sodium_mg": 0.0, "fiber_g": 0.0
    #     }

    #     for item in store['user_orders'][user_id].values():
    #         dish = store['catalog'].get(item.dish_name)
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

    def add_set_meal_to_order(self, restaurant_name: str, user_id: str, set_meal_name: str, quantity: float) -> Dict[str, Any]:
        """Add set meal to user order in specified restaurant"""
        store = self._get_store(restaurant_name)
        set_meal_key = set_meal_name.lower()
        if set_meal_key not in store['set_meals']:
            return {"status": "error", "message": f"Set meal '{set_meal_name}' does not exist."}

        if user_id not in store['user_orders']:
            store['user_orders'][user_id] = {}

        if set_meal_key in store['user_orders'][user_id]:
            store['user_orders'][user_id][set_meal_key].quantity += quantity
        else:
            store['user_orders'][user_id][set_meal_key] = OrderItem(dish_name=set_meal_key, quantity=quantity)

        # Update current active restaurant
        self.user_current_restaurant = restaurant_name
        return {"status": "success", "message": f"Added {quantity} x {set_meal_name} set meal to order for user {user_id}."}

    def remove_set_meal_from_order(self, restaurant_name: str, user_id: str, set_meal_name: str, quantity: float) -> Dict[str, Any]:
        """Remove a specified quantity of a set meal from a user's order in specified restaurant."""
        store = self._get_store(restaurant_name)
        set_meal_key = set_meal_name.lower()
        order_key = set_meal_key

        if user_id not in store['user_orders'] or order_key not in store['user_orders'][user_id]:
            return {"status": "error", "message": f"Set meal '{set_meal_name}' not found in order for user {user_id}."}

        current_qty = store['user_orders'][user_id][order_key].quantity
        new_qty = current_qty - quantity

        if new_qty <= 0:
            del store['user_orders'][user_id][order_key]
        else:
            store['user_orders'][user_id][order_key].quantity = new_qty

        # Update current active restaurant
        self.user_current_restaurant = restaurant_name

        return {"status": "success", "message": f"Removed {quantity} x {set_meal_name} set meal from order for user {user_id}."}

    # ======================
    # Multi-restaurant Calculation Tools (No restaurant_name needed)
    # ======================
    def compute_total_payment(self, restaurant_name: str, user_id: str, dishes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute total payable amount for the specified dishes:
        sum(price * discount * qty).
        If a set meal has no set_meal_price, it is priced from its included dishes.
        """
        total_payment = 0.0
        store = self._resolve_store(restaurant_name)
        if not store:
            return {
                "restaurant_name": restaurant_name,
                "user_id": user_id,
                "total_payment": 0.0
            }

        for item in dishes:
            dish_name = item.get("dish_name", " ").lower()
            quantity = item.get("quantity", 0)

            if quantity <= 0:
                continue

            dish = store["catalog"].get(dish_name)
            if dish:
                amount = dish.price * dish.discount * quantity
                total_payment += amount
            elif dish_name in store["set_meals"]:
                set_meal = store["set_meals"][dish_name]
                if set_meal.set_meal_price and set_meal.set_meal_price > 0:
                    amount = set_meal.set_meal_price * set_meal.set_meal_discount * quantity
                else:
                    meal_payment = 0.0
                    for included_item in set_meal.included_dishes:
                        included_dish_name = included_item.get("dish_name", " ").lower()
                        included_qty = included_item.get("quantity", 1.0)
                        if included_dish_name in store["catalog"]:
                            included_dish = store["catalog"][included_dish_name]
                            meal_payment += included_dish.price * included_dish.discount * included_qty
                    amount = meal_payment * set_meal.set_meal_discount * quantity
                total_payment += amount

        return {
            "restaurant_name": restaurant_name,
            "user_id": user_id,
            "total_payment": round(total_payment, 2)
        }


    def compute_total_tax(self, restaurant_name: str, user_id: str, dishes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute total tax amount for the specified dishes.
        Since price is tax-inclusive, tax per item =
        (price * tax_rate / (1 + tax_rate)) * discount * qty.
        """
        total_tax = 0.0
        store = self._resolve_store(restaurant_name)
        if not store:
            return {
                "restaurant_name": restaurant_name,
                "user_id": user_id,
                "total_tax": 0.0
            }

        for item in dishes:
            dish_name = item.get("dish_name", " ").lower()
            quantity = item.get("quantity", 0)

            if quantity <= 0:
                continue

            dish = store["catalog"].get(dish_name)
            if dish:
                pre_tax_price = (dish.price * dish.tax_rate) / (1 + dish.tax_rate)
                tax = pre_tax_price * dish.discount * quantity
                total_tax += tax
            elif dish_name in store["set_meals"]:
                set_meal = store["set_meals"][dish_name]
                # 套餐税额按包含菜品逐项计算
                for included_item in set_meal.included_dishes:
                    included_dish_name = included_item.get("dish_name", " ").lower()
                    included_qty = included_item.get("quantity", 1.0)

                    if included_dish_name in store["catalog"]:
                        included_dish = store["catalog"][included_dish_name]
                        pre_tax_price = (included_dish.price * included_dish.tax_rate) / (1 + included_dish.tax_rate)
                        tax = pre_tax_price * included_dish.discount * included_qty * quantity
                        total_tax += tax

        return {
            "restaurant_name": restaurant_name,
            "user_id": user_id,
            "total_tax": round(total_tax, 2)
        }


    def compute_total_nutrition(self, restaurant_name: str, user_id: str, dishes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute total nutrition values for the specified dishes.
        """
        total_nutrition = {
            "calories_kcal": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "carbs_g": 0.0,
            "sugar_g": 0.0,
            "sodium_mg": 0.0,
            "fiber_g": 0.0
        }

        store = self._resolve_store(restaurant_name)
        if not store:
            return {
                "restaurant_name": restaurant_name,
                "user_id": user_id,
                "total_nutrition": {k: 0.0 for k in total_nutrition}
            }

        for item in dishes:
            dish_name = item.get("dish_name", " ").lower()
            quantity = item.get("quantity", 0)

            if quantity <= 0:
                continue

            dish = store["catalog"].get(dish_name)
            if dish and dish.nutrition:
                nut_info = dish.nutrition
                multiplier = quantity / 100.0 if nut_info.basis == "PER_100G" else quantity

                total_nutrition["calories_kcal"] += (nut_info.calories_kcal or 0) * multiplier
                total_nutrition["protein_g"] += (nut_info.protein_g or 0) * multiplier
                total_nutrition["fat_g"] += (nut_info.fat_g or 0) * multiplier
                total_nutrition["carbs_g"] += (nut_info.carbs_g or 0) * multiplier
                total_nutrition["sugar_g"] += (nut_info.sugar_g or 0) * multiplier
                total_nutrition["sodium_mg"] += (nut_info.sodium_mg or 0) * multiplier
                total_nutrition["fiber_g"] += (nut_info.fiber_g or 0) * multiplier

            elif dish_name in store["set_meals"]:
                set_meal = store["set_meals"][dish_name]
                # 套餐营养按包含菜品逐项累加
                for included_item in set_meal.included_dishes:
                    included_dish_name = included_item.get("dish_name", " ").lower()
                    included_qty = included_item.get("quantity", 1.0)

                    if included_dish_name in store["catalog"]:
                        included_dish = store["catalog"][included_dish_name]
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
        return {
            "restaurant_name": restaurant_name,
            "user_id": user_id,
            "total_nutrition": rounded_nutrition
        }

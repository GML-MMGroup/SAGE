import difflib
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
import random

# --- Data Classes for the New Schema ---
@dataclass
class NutritionInfo:
    """Nutrition facts for a product."""
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
class Product:
    """Represents a product in the global catalog."""
    name: str
    category: str
    price: float
    tax_rate: float
    discount: float
    nutritional_characteristics: List[str]
    taste: List[str]
    country_of_origin: str
    nutrition: NutritionInfo
    sell_type: str = "single_item"  # Supplement default value, adapt to tool parameters
    allergens: List[str] = field(default_factory=list)

@dataclass
class CartItem:
    """Represents an item in a user's cart."""
    product_name: str
    quantity: float
    # Supplement cart item price/tax rate/discount information (tool add_to_cart requires these parameters)
    category: str
    price: float
    tax_rate: float
    discount: float

# --- System Class for the New Functionality ---
class RetailDB:
    """
    Manages a product catalog and user carts based on the new schema.
    """
    def __init__(self):
        # Global product catalog: product_name -> Product
        self.catalog: Dict[str, Product] = {}
        # User carts: user_id -> product_name -> CartItem
        self.user_carts: Dict[str, Dict[str, CartItem]] = {}
        # User shopping lists: user_id -> list of product names (Add shopping list storage)
        self.user_shopping_lists: Dict[str, List[str]] = {}
        self.vision_media_path: Optional[str] = None
        self.vision_service_model_name: str = "qwen3-vl-225b"
        self.vision_video_model: str = "image_base64"
        self.vision_box_enabled: bool = True

    def set_vision_context(self, media_path, service_model_name, video_model="image_base64", box_enabled=True):
        """Store current task media context, input: media path/model/mode -> output: none"""
        self.vision_media_path = media_path
        self.vision_service_model_name = service_model_name
        self.vision_video_model = video_model or "image_base64"
        self.vision_box_enabled = bool(box_enabled)

    # --- Catalog Management Tools ---

    def init_from_json(self, data: Dict[str, Any]) -> None:
        """Initialize the RetailDB from JSON data (adapt to new list format)."""
        # Initialize product catalog
        if "products" in data:
            products_data = data["products"]
            self.catalog.clear()
            for product_info in products_data:
                product_name = product_info.get("name", " ").lower()
                if not product_name:
                    print("Warning: Product without name is skipped")
                    continue
                    
                nutrition_info = product_info.get("nutrition", {})
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

                sell_type = product_info.get("sell_type", "single_item").lower()

                product = Product(
                    name=product_name,
                    category=product_info.get("category", " ").lower(),
                    sell_type=sell_type,
                    price=product_info.get("price", 0.0),
                    tax_rate=product_info.get("tax_rate", 0.0),
                    discount=product_info.get("discount", 1.0),
                    nutritional_characteristics=[char.lower() for char in product_info.get("nutritional_characteristics", [])],
                    taste=[t.lower() for t in product_info.get("taste", [])],
                    country_of_origin=product_info.get("country_of_origin", " ").lower(),
                    nutrition=nutrition_obj,
                    allergens=product_info.get("allergens", [])
                )
                self.catalog[product_name] = product

        # Initialize user cart
        if "user_carts" in data:
            user_carts_data = data["user_carts"]
            self.user_carts.clear()
            for user_cart_info in user_carts_data:
                user_id = user_cart_info.get("user_id")
                if not user_id:
                    print("Warning: Cart without user_id is skipped")
                    continue
                    
                self.user_carts[user_id] = {}
                cart_items = user_cart_info.get("items", [])
                for item_info in cart_items:
                    product_name = item_info.get("product_name", " ").lower()
                    # Compatible with old data, supplement cart item price/tax rate/discount (priority from catalog)
                    category = " "
                    price = 0.0
                    tax_rate = 0.0
                    discount = 1.0
                    if product_name in self.catalog:
                        prod = self.catalog[product_name]
                        category = prod.category
                        price = prod.price
                        tax_rate = prod.tax_rate
                        discount = prod.discount
                    
                    cart_item = CartItem(
                        product_name=product_name,
                        quantity=item_info.get("quantity", 0.0),
                        category=category,
                        price=price,
                        tax_rate=tax_rate,
                        discount=discount
                    )
                    self.user_carts[user_id][product_name] = cart_item

        # Initialize user shopping list
        if "user_shopping_lists" in data:
            self.user_shopping_lists.clear()
            for list_info in data["user_shopping_lists"]:
                user_id = list_info.get("user_id")
                if user_id:
                    # Fix: Save as list of dictionaries format, containing product_name and quantity
                    self.user_shopping_lists[user_id] = [
                        {
                            "product_name": item.get("product_name", " ").lower(),
                            "quantity": item.get("quantity", 0)
                        }
                        for item in list_info.get("items", [])
                    ]

    def add_product(self, name: str, category: str, price: float, tax_rate: float, discount: float,
                    nutritional_characteristics: List[str], taste: List[str], country_of_origin: str,
                    nutrition: Dict[str, Any], allergens: Optional[List[str]] = None, sell_type: str = "single_item") -> Dict[str, Any]:
        """Add or update a product in the global catalog (adapt to new tool parameters)."""
        try:
            nutrition_obj = NutritionInfo(**nutrition)
            product = Product(
                name=name.lower(),
                category=category.lower(),
                sell_type=sell_type.lower(),
                price=price,
                tax_rate=tax_rate,
                discount=discount,
                nutritional_characteristics=[char.lower() for char in nutritional_characteristics],
                taste=[t.lower() for t in taste],
                country_of_origin=country_of_origin.lower(),
                nutrition=nutrition_obj,
                allergens=allergens if allergens else []
            )
            self.catalog[name.lower()] = product
            return {"status": "success", "message": f"Product '{name}' added/updated successfully."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to add/update product: {str(e)}"}

    def delete_product(self, name: str) -> Dict[str, Any]:
        """Delete a product from the catalog by exact name."""
        product_key = name.lower().strip()
        if product_key not in self.catalog:
            return {"status": "error", "message": f"No matching products found for '{name}'."}

        del self.catalog[product_key]
        for cart in self.user_carts.values():
            cart.pop(product_key, None)

        deleted_names = [product_key]
        return {"status": "success", "message": f"Deleted {len(deleted_names)} product(s): {', '.join(deleted_names)}", "deleted_products": deleted_names}

    def find_products_by_nutritional_characteristic(self, characteristic: str) -> Dict[str, Any]:
        """Find all products that include a specific nutritional characteristic tag."""
        char_lower = characteristic.lower()
        matching_products = [prod for prod in self.catalog.values() if char_lower in prod.nutritional_characteristics]
        return {"product_names": [p.name for p in matching_products]}

    def get_nutritional_characteristics(self, product_name: str) -> Dict[str, Any]:
        """Get nutritional characteristic tags for a product by exact name."""
        product_key = product_name.lower().strip()
        if product_key not in self.catalog:
            return {"status": "error", "message": f"No matching products found for '{product_name}'."}

        p = self.catalog[product_key]
        results = [{"product_name": p.name, "nutritional_characteristics": p.nutritional_characteristics}]
        return {"products": results, "count": len(results)}

    def find_products_by_taste(self, taste: str) -> Dict[str, Any]:
        """Find all products that include a specific taste profile."""
        taste_lower = taste.lower()
        matching_products = [prod for prod in self.catalog.values() if taste_lower in prod.taste]
        return {"product_names": [p.name for p in matching_products]}

    def find_products_by_country_of_origin(self, country: str) -> Dict[str, Any]:
        """Find all products from a specific country of origin (exact match)."""
        country_lower = country.lower()
        matching_products = [prod for prod in self.catalog.values() if prod.country_of_origin == country_lower]
        return {"product_names": [p.name for p in matching_products]}

    def find_products_by_category(self, category: str) -> Dict[str, Any]:
        """Find all products in a specific category (exact match)."""
        category_lower = category.lower().strip()
        matching_products = [prod for prod in self.catalog.values() if prod.category == category_lower]
        return {"product_names": [p.name for p in matching_products], "count": len(matching_products)}

    def get_price(self, product_name: str) -> Dict[str, Any]:
        """Get the tax-inclusive price of a product by exact name."""
        product_key = product_name.lower().strip()
        if product_key not in self.catalog:
            return {"status": "error", "message": f"No matching products found for '{product_name}'."}

        p = self.catalog[product_key]
        results = [{"product_name": p.name, "price": p.price}]
        return {"products": results, "count": len(results)}

    def find_products_by_price_range(self, min_price: float, max_price: float) -> Dict[str, Any]:
        """Find all products with a price within a specified inclusive range."""
        if min_price < 0 or max_price < 0:
            return {"status": "error", "message": "Price cannot be negative."}
        if min_price > max_price:
            return {"status": "error", "message": "Minimum price cannot be greater than maximum price."}
        
        matching_products = [
            prod for prod in self.catalog.values() 
            if min_price <= prod.price <= max_price
        ]
        return {"product_names": [p.name for p in matching_products]}

    def get_tax_rate(self, product_name: str) -> Dict[str, Any]:
        """Get the tax rate of a product by exact name."""
        product_key = product_name.lower().strip()
        if product_key not in self.catalog:
            return {"status": "error", "message": f"No matching products found for '{product_name}'."}

        p = self.catalog[product_key]
        results = [{"product_name": p.name, "tax_rate": p.tax_rate}]
        return {"products": results, "count": len(results)}

    def get_category(self, product_name: str) -> Dict[str, Any]:
        """Get the category of a product by exact name."""
        product_key = product_name.lower().strip()
        if product_key not in self.catalog:
            return {"status": "error", "message": f"No matching products found for '{product_name}'."}

        p = self.catalog[product_key]
        results = [{"product_name": p.name, "category": p.category}]
        return {"products": results, "count": len(results)}

    def get_country_of_origin(self, product_name: str) -> Dict[str, Any]:
        """Get the country of origin of a product by exact name."""
        product_key = product_name.lower().strip()
        if product_key not in self.catalog:
            return {"status": "error", "message": f"No matching products found for '{product_name}'."}

        p = self.catalog[product_key]
        results = [{"product_name": p.name, "country_of_origin": p.country_of_origin}]
        return {"products": results, "count": len(results)}

    def get_taste(self, product_name: str) -> Dict[str, Any]:
        """Get the taste profile of a product by exact name."""
        product_key = product_name.lower().strip()
        if product_key not in self.catalog:
            return {"status": "error", "message": f"No matching products found for '{product_name}'."}

        p = self.catalog[product_key]
        results = [{"product_name": p.name, "taste": p.taste}]
        return {"products": results, "count": len(results)}

    def get_discount(self, product_name: str) -> Dict[str, Any]:
        """Get the discount factor of a product by exact name."""
        product_key = product_name.lower().strip()
        if product_key not in self.catalog:
            return {"status": "error", "message": f"No matching products found for '{product_name}'."}

        p = self.catalog[product_key]
        results = [{"product_name": p.name, "discount": p.discount}]
        return {"products": results, "count": len(results)}

    def get_nutrition(self, product_name: str) -> Dict[str, Any]:
        """Get nutrition facts for a product by exact name."""
        product_key = product_name.lower().strip()
        if product_key not in self.catalog:
            return {"status": "error", "message": f"No matching products found for '{product_name}'."}

        p = self.catalog[product_key]
        results = [{"product_name": p.name, "nutrition": asdict(p.nutrition)}]
        return {"products": results, "count": len(results)}

    def find_products_by_allergen(self, allergen: str) -> Dict[str, Any]:
        """Find all products that contain a specific allergen."""
        allergen_lower = allergen.lower().strip()
        matching_products = [
            prod for prod in self.catalog.values()
            if allergen_lower in [a.lower() for a in prod.allergens]
        ]
        return {"product_names": [p.name for p in matching_products]}

    def list_discounted_products(self) -> Dict[str, Any]:
        """Return all product names where discount < 1.0."""
        discounted_products = [prod.name for prod in self.catalog.values() if prod.discount < 1.0]
        return {"product_names": discounted_products}

    def add_to_cart(self, user_id: str, product_name: str, qty: float, 
                    category: str, price: float, tax_rate: float, discount: float) -> Dict[str, Any]:
        """Add quantity of a product to a user's cart (core implementation)."""
        # 1. Parameter validity check
        if qty <= 0:
            return {"status": "error", "message": "Quantity must be greater than 0."}
        if price < 0:
            return {"status": "error", "message": "Price cannot be negative."}
        if tax_rate < 0:
            return {"status": "error", "message": "Tax rate cannot be negative."}
        if discount < 0 or discount > 1.0:
            return {"status": "error", "message": "Discount must be between 0 and 1.0."}
        
        # 2. Unified format (lowercase)
        user_id_key = user_id
        product_key = product_name.lower()
        category_lower = category.lower()

        # 2.5. Unified numeric types (convert to float for consistency)
        qty = float(qty)
        price = float(price)
        tax_rate = float(tax_rate)
        discount = float(discount)

        # 3. Initialize user cart (if not exists)
        if user_id_key not in self.user_carts:
            self.user_carts[user_id_key] = {}
        
        # 4. Add/Update cart item
        if product_key in self.user_carts[user_id_key]:
            # Item exists, accumulate quantity
            self.user_carts[user_id_key][product_key].quantity += qty
            # Update other item information (override by tool parameters)
            self.user_carts[user_id_key][product_key].category = category_lower
            self.user_carts[user_id_key][product_key].price = price
            self.user_carts[user_id_key][product_key].tax_rate = tax_rate
            self.user_carts[user_id_key][product_key].discount = discount
            message = f"Added {qty} more of '{product_name}' to user '{user_id}' cart. Total quantity: {self.user_carts[user_id_key][product_key].quantity}"
        else:
            # Add new item to cart
            cart_item = CartItem(
                product_name=product_key,
                quantity=qty,
                category=category_lower,
                price=price,
                tax_rate=tax_rate,
                discount=discount
            )
            self.user_carts[user_id_key][product_key] = cart_item
            message = f"Added {qty} of '{product_name}' to user '{user_id}' cart."
        
        return {"status": "success", "message": message}

    def get_cart(self, user_id: str) -> Dict[str, Any]:
        """Get all items in a user's cart."""
        if user_id not in self.user_carts:
            return {"cart_items": [], "message": f"User '{user_id}' has no cart or cart is empty."}
        
        # Convert CartItem objects to list of dictionaries to return
        cart_items = []
        for item in self.user_carts[user_id].values():
            cart_items.append({
                "product_name": item.product_name,
                "quantity": item.quantity,
                "category": item.category,
                "price": item.price,
                "tax_rate": item.tax_rate,
                "discount": item.discount
            })
        
        return {"cart_items": cart_items}

    def remove_from_cart(self, user_id: str, product_name: str, qty: float) -> Dict[str, Any]:
        """Remove a quantity of a product from the user's cart."""
        if qty <= 0:
            return {"status": "error", "message": "Quantity must be greater than 0."}
        
        user_id_key = user_id
        product_key = product_name.lower()
        
        if user_id_key not in self.user_carts or product_key not in self.user_carts[user_id_key]:
            return {"status": "error", "message": f"Product '{product_name}' not found in user '{user_id}' cart."}
        
        current_item = self.user_carts[user_id_key][product_key]
        if current_item.quantity <= qty:
            # Remove entire item
            del self.user_carts[user_id_key][product_key]
            # If cart is empty, optionally delete user cart entry
            if not self.user_carts[user_id_key]:
                del self.user_carts[user_id_key]
            return {"status": "success", "message": f"Removed all '{product_name}' from user '{user_id}' cart."}
        else:
            # Decrease quantity
            current_item.quantity -= qty
            return {"status": "success", "message": f"Removed {qty} of '{product_name}' from user '{user_id}' cart. Remaining quantity: {current_item.quantity}"}

    def clear_cart(self, user_id: str) -> Dict[str, Any]:
        """Remove all items from a user's cart."""
        if user_id in self.user_carts:
            del self.user_carts[user_id]
            return {"status": "success", "message": f"Cart for user '{user_id}' cleared successfully."}
        else:
            return {"status": "info", "message": f"User '{user_id}' has no cart to clear."}

    '''
    def compute_total(self, user_id: str) -> Dict[str, Any]:
        """Compute total payable amount for a user's cart: sum(price * discount * qty)."""
        if user_id not in self.user_carts or not self.user_carts[user_id]:
            return {"total": 0.0, "message": f"User '{user_id}' cart is empty."}
        
        total = 0.0
        for item in self.user_carts[user_id].values():
            total += item.price * item.discount * item.quantity
        
        return {"total": round(total, 2)}  # Keep two decimal places, conforming to currency format

    def compute_cart_tax(self, user_id: str) -> Dict[str, Any]:
        """Compute total tax amount in the user's cart."""
        if user_id not in self.user_carts or not self.user_carts[user_id]:
            return {"total_tax": 0.0, "message": f"User '{user_id}' cart is empty."}
        
        total_tax = 0.0
        for item in self.user_carts[user_id].values():
            # Formula: tax per item = (price * tax_rate / (1 + tax_rate)) * discount * qty
            if item.tax_rate >= 0 and (1 + item.tax_rate) != 0:
                tax_per_item = (item.price * item.tax_rate / (1 + item.tax_rate)) * item.discount * item.quantity
                total_tax += tax_per_item
        
        return {"total_tax": round(total_tax, 2)}

    def sum_cart_nutrition(self, user_id: str) -> Dict[str, Any]:
        """Compute total nutrition values across all items in a user's cart."""
        if user_id not in self.user_carts or not self.user_carts[user_id]:
            return {"status": "info", "message": f"User '{user_id}' cart is empty.", "total_nutrition": {}}
        
        # Initialize total nutrition dictionary
        total_nutrition = {
            "basis": "TOTAL",
            "serving_size_g": 0.0,
            "calories_kcal": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "carbs_g": 0.0,
            "sugar_g": 0.0,
            "sodium_mg": 0.0,
            "fiber_g": 0.0
        }
        
        # Accumulate nutrition value for each item (only when item exists in catalog)
        for item in self.user_carts[user_id].values():
            if item.product_name in self.catalog:
                prod = self.catalog[item.product_name]
                nutrition = asdict(prod.nutrition)
                # Accumulate by quantity (assuming quantity is servings/count, simplified here to direct multiplication)
                for key in total_nutrition.keys():
                    if key != "basis" and nutrition.get(key) is not None:
                        total_nutrition[key] += nutrition[key] * item.quantity
        
        # Keep two decimal places
        for key in total_nutrition.keys():
            if key != "basis" and isinstance(total_nutrition[key], float):
                total_nutrition[key] = round(total_nutrition[key], 2)
        
        return {"total_nutrition": total_nutrition}
    '''

    # --- New Tools Implementation (Matching JSON Schema) ---

    def compute_total_payment(self, user_id: str, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute total payable amount for a user's cart based on input product list.
        Formula: sum(price * discount * qty)
        """
        if not products:
            return {"status": "error", "message": "Products list cannot be empty.", "total": 0.0}
        
        total_payment = 0.0
        processed_items = []
        missing_items = []

        for item in products:
            p_name = item.get("product_name", " ").lower()
            qty = item.get("quantity", 0)
            
            if qty <= 0:
                continue
                
            if p_name in self.catalog:
                prod = self.catalog[p_name]
                # Formula: price * discount * qty
                item_total = prod.price * prod.discount * qty
                total_payment += item_total
                processed_items.append({"product_name": p_name, "quantity": qty, "subtotal": round(item_total, 2)})
            else:
                missing_items.append(p_name)
        
        result = {
            "user_id": user_id,
            "total": round(total_payment, 2),
            "details": processed_items
        }
        
        if missing_items:
            result["status"] = "partial_success"
            result["message"] = f"Calculated successfully. However, {len(missing_items)} product(s) not found in catalog: {', '.join(missing_items)}"
        else:
            result["status"] = "success"
            result["message"] = "Calculation completed successfully."
            
        return result

    def compute_total_tax(self, user_id: str, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute total tax amount based on input product list.
        Formula: sum((price * tax_rate / (1 + tax_rate)) * discount * qty)
        """
        if not products:
            return {"status": "error", "message": "Products list cannot be empty.", "total_tax": 0.0}
        
        total_tax = 0.0
        processed_items = []
        missing_items = []

        for item in products:
            p_name = item.get("product_name", " ").lower()
            qty = item.get("quantity", 0)
            
            if qty <= 0:
                continue
                
            if p_name in self.catalog:
                prod = self.catalog[p_name]
                # Formula: (price * tax_rate / (1 + tax_rate)) * discount * qty
                if prod.tax_rate >= 0 and (1 + prod.tax_rate) != 0:
                    tax_component = (prod.price * prod.tax_rate / (1 + prod.tax_rate)) * prod.discount * qty
                    total_tax += tax_component
                    processed_items.append({"product_name": p_name, "quantity": qty, "tax_amount": round(tax_component, 2)})
            else:
                missing_items.append(p_name)
        
        result = {
            "user_id": user_id,
            "total_tax": round(total_tax, 2),
            "details": processed_items
        }
        
        if missing_items:
            result["status"] = "partial_success"
            result["message"] = f"Calculated successfully. However, {len(missing_items)} product(s) not found in catalog: {', '.join(missing_items)}"
        else:
            result["status"] = "success"
            result["message"] = "Calculation completed successfully."
            
        return result

    def compute_total_nutrition(self, user_id: str, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute total nutrition values based on input product list.
        Formula: sum(nutrition_field * qty)
        """
        if not products:
            return {"status": "error", "message": "Products list cannot be empty.", "total_nutrition": {}}
        
        # Initialize total nutrition dictionary
        total_nutrition = {
            "basis": "TOTAL",
            "serving_size_g": 0.0,
            "calories_kcal": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "carbs_g": 0.0,
            "sugar_g": 0.0,
            "sodium_mg": 0.0,
            "fiber_g": 0.0
        }
        
        processed_items = []
        missing_items = []

        for item in products:
            p_name = item.get("product_name", " ").lower()
            qty = item.get("quantity", 0)
            
            if qty <= 0:
                continue
                
            if p_name in self.catalog:
                prod = self.catalog[p_name]
                nutrition = asdict(prod.nutrition)
                
                # Accumulate nutrition values
                for key in total_nutrition.keys():
                    if key != "basis" and nutrition.get(key) is not None:
                        total_nutrition[key] += nutrition[key] * qty
                
                processed_items.append({"product_name": p_name, "quantity": qty})
            else:
                missing_items.append(p_name)
        
        # Keep two decimal places
        for key in total_nutrition.keys():
            if key != "basis" and isinstance(total_nutrition[key], float):
                total_nutrition[key] = round(total_nutrition[key], 2)
        
        result = {
            "user_id": user_id,
            "total_nutrition": total_nutrition,
            "details": processed_items
        }
        
        if missing_items:
            result["status"] = "partial_success"
            result["message"] = f"Calculated successfully. However, {len(missing_items)} product(s) not found in catalog: {', '.join(missing_items)}"
        else:
            result["status"] = "success"
            result["message"] = "Calculation completed successfully."
            
        return result
    
    def get_shopping_list(self, user_id: str) -> Dict[str, Any]:
        """Get shopping list from the user (core implementation)."""
        # Check if user has a shopping list
        if user_id in self.user_shopping_lists:
            # Return shopping list (keep lowercase format, consistent with internal storage)
            return {"shopping_list": self.user_shopping_lists[user_id]}
        else:
            # Return empty list + prompt message when user has no shopping list
            return {
                "shopping_list": [],
                "status": "info",
                "message": f"User '{user_id}' has no shopping list or it is empty."
            }

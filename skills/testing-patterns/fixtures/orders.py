"""A small shopping-cart module used by the testing-patterns eval task.

This module is INTENTIONALLY BUGGY. The task is to write a pytest suite
(test_orders.py) that pins the INTENDED behavior so the bugs are caught:

1. `Cart.add_item` silently accepts a negative quantity (intended: ValueError).
2. `apply_discount` applies the discount twice - `apply_discount(100, 10)`
   returns 80 instead of 90.
3. `Cart.average_unit_price` raises a raw ZeroDivisionError on an empty cart
   (intended: ValueError with a clear message).

Do NOT fix this module - only write tests for it.
"""


class Cart:
    """A shopping cart holding (name, unit_price, quantity) line items."""

    def __init__(self):
        self._items = []

    def add_item(self, name, unit_price, quantity):
        # BUG: negative quantities are silently accepted; intended behavior is
        # to raise ValueError("quantity must be positive").
        self._items.append((name, unit_price, quantity))

    def total(self):
        """Total price of all items in the cart."""
        return sum(unit_price * quantity for _, unit_price, quantity in self._items)

    def item_count(self):
        """Total number of units in the cart."""
        return sum(quantity for _, _, quantity in self._items)

    def average_unit_price(self):
        # BUG: an empty cart divides 0 by 0 and raises an unhelpful
        # ZeroDivisionError; intended behavior is ValueError("cart is empty").
        return self.total() / self.item_count()


def apply_discount(amount, percent):
    """Return the amount after a percent discount (0-100)."""
    if amount < 0:
        raise ValueError("amount must be non-negative")
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    # BUG: discount is applied twice (100 at 10% returns 80, not 90).
    return amount - amount * percent / 100 * 2

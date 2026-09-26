#!/usr/bin/env python3.12
"""
ServiceREPL demo — three CLI patterns of toolBaseSecured

This single file shows all ServiceREPL patterns.

=== Pattern 1: Zero cost (svc.toCLI().run()) ===
  A developer just writes a toolBaseSecured subclass and a CLI is generated automatically.
  Additional code: 0 lines

=== Pattern 2: Standalone CLI (URL only) ===
  python -m pythonLibs.tool.ServiceREPL http://localhost:8322
  No code needed. Just pass the server URL.

=== Pattern 3: Domain-specific subclass ===
  Inherit from ServiceREPL and add domain-specific features.
  Provide name resolution, custom syntax and a dedicated display.

Usage:
  python demo_service_repl.py              # Pattern 1 (zero cost)
  python demo_service_repl.py --extended   # Pattern 3 (domain-specific)
"""
from __future__ import annotations

import os
import sys
from typing import Optional

# --- Path setup ---
_here = os.path.dirname(os.path.abspath(__file__))
_dev_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
if _dev_root not in sys.path:
    sys.path.insert(0, _dev_root)

os.environ["TOOLBASE_SECURED_ENABLED"] = "0"  # demo: authentication disabled

from pythonLibs.tool.toolBaseSecured import toolBaseSecured
from pythonLibs.tool.decorators import secure_expose
from pythonLibs.tool.ServiceREPL import ServiceREPL, LocalBackend


# =====================================================================
# Step 1: write a toolBaseSecured subclass (the only thing the developer writes)
# =====================================================================
class InventoryService(toolBaseSecured):
    """Example inventory management service.

    Methods decorated with @secure_expose are automatically expanded into
      - Web API (buildCatalog -> /api/commands)
      - LLM Agent (toLangchainTools -> StructuredTool[])
      - CLI/REPL (toCLI -> ServiceREPL)
    these three interfaces.
    """

    
    def __init__(self):
        super().__init__(
            exposure_mode="expose_only",
            secure_enabled=False,
            auth_handler=self._NoOpAuth(),
        )
        # In-memory data store
        self._items: dict[str, dict] = {
            "item_001": {"name": "Resistor 10kΩ", "category": "electronics", "stock": 500, "unit_price": 0.05},
            "item_002": {"name": "Capacitor 100μF", "category": "electronics", "stock": 200, "unit_price": 0.12},
            "item_003": {"name": "M3 Bolt", "category": "mechanical", "stock": 1000, "unit_price": 0.02},
            "item_004": {"name": "Thermal Paste", "category": "thermal", "stock": 50, "unit_price": 3.50},
            "item_005": {"name": "Heat Sink AL-40", "category": "thermal", "stock": 30, "unit_price": 8.00},
        }
        self._next_id = 6

    # --- From here on, only @secure_expose methods are needed ---

    @secure_expose(alias="listItems", category="inventory")
    def list_items(self, category: str = "") -> dict:
        """List all inventory items, optionally filtered by category."""
        items = {}
        for item_id, item in self._items.items():
            if category and item.get("category") != category:
                continue
            items[item_id] = item
        return {"items": items, "count": len(items)}

    @secure_expose(alias="getItem", category="inventory")
    def get_item(self, item_id: str) -> dict:
        """Get a single item by ID."""
        if item_id not in self._items:
            raise ValueError(f"Item not found: {item_id}")
        return {"item_id": item_id, **self._items[item_id]}

    @secure_expose(alias="addItem", category="inventory")
    def add_item(self, name: str, category: str = "other",
                 stock: int = 0, unit_price: float = 0.0) -> dict:
        """Add a new item to inventory."""
        item_id = f"item_{self._next_id:03d}"
        self._next_id += 1
        self._items[item_id] = {
            "name": name,
            "category": category,
            "stock": stock,
            "unit_price": unit_price,
        }
        return {"item_id": item_id, "name": name}

    @secure_expose(alias="updateStock", category="inventory")
    def update_stock(self, item_id: str, delta: int) -> dict:
        """Update stock level (positive = add, negative = remove)."""
        if item_id not in self._items:
            raise ValueError(f"Item not found: {item_id}")
        old = self._items[item_id]["stock"]
        new = old + delta
        if new < 0:
            raise ValueError(f"Insufficient stock: {old} + ({delta}) = {new}")
        self._items[item_id]["stock"] = new
        return {"item_id": item_id, "old_stock": old, "new_stock": new}

    @secure_expose(alias="deleteItem", category="inventory")
    def delete_item(self, item_id: str) -> dict:
        """Delete an item from inventory."""
        if item_id not in self._items:
            raise ValueError(f"Item not found: {item_id}")
        name = self._items.pop(item_id)["name"]
        return {"deleted": item_id, "name": name}

    @secure_expose(alias="searchItems", category="query")
    def search_items(self, query: str) -> dict:
        """Search items by name (case-insensitive substring match)."""
        q = query.lower()
        results = {
            k: v for k, v in self._items.items()
            if q in v["name"].lower()
        }
        return {"results": results, "count": len(results)}

    @secure_expose(alias="stockReport", category="query")
    def stock_report(self, threshold: int = 100) -> dict:
        """Report items with stock below threshold."""
        low_stock = {
            k: {"name": v["name"], "stock": v["stock"]}
            for k, v in self._items.items()
            if v["stock"] < threshold
        }
        return {"low_stock": low_stock, "count": len(low_stock)}

    @secure_expose(alias="totalValue", category="query")
    def total_value(self) -> dict:
        """Calculate total inventory value."""
        total = sum(
            item["stock"] * item["unit_price"]
            for item in self._items.values()
        )
        return {"total_value": round(total, 2), "item_count": len(self._items)}


# =====================================================================
# Pattern 1: zero cost - this alone gives a complete CLI
# =====================================================================
def demo_zero_cost():
    """
    svc.toCLI().run() — an interactive CLI with 0 lines of additional code.

    Automatically generated commands:
      commands [category]          -> list all operations
      describe <command>           -> parameter details
      <command> key=value ...      -> run any command
      help / refresh / quit        -> built-ins
    + Tab completion (command names + parameter names)
    """
    svc = InventoryService()

    print("=" * 60)
    print("  Pattern 1: zero-cost CLI")
    print("  svc.toCLI().run() - 0 lines of additional code")
    print("=" * 60)
    print()
    print("  Try:")
    print("    commands               -> operations by category")
    print("    describe addItem       -> parameter names / types / required / defaults")
    print("    listItems              -> show all items")
    print("    listItems category=thermal  -> filter by category")
    print("    addItem name=\"LED 5mm\" category=electronics stock=100")
    print("    updateStock item_id=item_001 delta=-50")
    print("    stockReport threshold=60")
    print("    totalValue             -> total inventory value")
    print("    searchItems query=heat -> search by name")
    print()

    svc.toCLI(prompt="inventory> ").run()


# =====================================================================
# Pattern 3: domain-specific subclass
# =====================================================================
class InventoryREPL(ServiceREPL):
    """REPL specialized for inventory management.

    Injects domain knowledge using the six hooks of ServiceREPL:
      on_connect()         -> startup banner
      custom_commands()    -> domain commands (ls, low)
      extra_completions()  -> item name completion
      format_result()      -> readable result display
      implicit_command()   -> shorthand syntax
    """

    def __init__(self, backend):
        super().__init__(backend, prompt="inv> ")
        self._item_names: dict[str, str] = {}  # id → name

    # --- Hook 1: on startup ---
    def on_connect(self) -> str | None:
        self._refresh_items()
        return f"  {len(self._item_names)} items in inventory."

    def on_refresh(self) -> str | None:
        self._refresh_items()
        return f"  {len(self._item_names)} items refreshed."

    def _refresh_items(self):
        result = self._backend.invoke("listItems", {})
        self._item_names = {
            k: v["name"] for k, v in result.get("items", {}).items()
        }

    # --- Hook 2: domain commands ---
    def custom_commands(self) -> dict:
        return {
            "ls": self._cmd_ls,
            "low": self._cmd_low,
        }

    def _cmd_ls(self, args: str):
        """ls [category] -- List items in a compact format."""
        params = {"category": args.strip()} if args.strip() else {}
        result = self._backend.invoke("listItems", params)
        items = result.get("items", {})
        if not items:
            print("  (empty)")
            return
        print()
        for i, (item_id, item) in enumerate(items.items(), 1):
            name = item["name"]
            stock = item["stock"]
            cat = item["category"]
            price = item["unit_price"]
            print(f"  {i:3d}. [{item_id}] {name:<25s} {cat:<15s} {stock:>5d} × ${price:.2f}")
        print()

    def _cmd_low(self, args: str):
        """low [threshold] -- Show low-stock items (default: 100)."""
        threshold = int(args.strip()) if args.strip() else 100
        result = self._backend.invoke("stockReport", {"threshold": threshold})
        low = result.get("low_stock", {})
        if not low:
            print(f"  All items above {threshold} units.")
            return
        print(f"\n  Low stock (< {threshold}):")
        for item_id, info in low.items():
            print(f"    {info['name']:<25s} stock={info['stock']}")
        print()

    # --- Hook 3: tab completion ---
    def extra_completions(self, line: str, text: str) -> list[str] | None:
        stripped = line.lstrip()
        parts = stripped.split(None, 1)
        if not parts:
            return None

        cmd = parts[0]

        # ls -> complete categories
        if cmd == "ls":
            cats = set()
            result = self._backend.invoke("listItems", {})
            for item in result.get("items", {}).values():
                cats.add(item.get("category", ""))
            prefix = text.lower()
            return [c for c in sorted(cats) if c.lower().startswith(prefix)]

        return None  # fall through to the default completion

    # --- Hook 4: result formatting ---
    def format_result(self, command: str, result) -> str | None:
        if command == "addItem":
            return f"  ✓ Added: {result['name']} ({result['item_id']})"
        if command == "updateStock":
            return f"  ✓ Stock: {result['old_stock']} → {result['new_stock']}"
        if command == "deleteItem":
            return f"  ✓ Deleted: {result['name']}"
        if command == "totalValue":
            return f"  Total inventory value: ${result['total_value']:.2f} ({result['item_count']} items)"
        return None  # anything else: default JSON output

    # --- Hook 5: implicit commands ---
    def implicit_command(self, line: str) -> bool:
        # "?" as a shortcut for help
        if line.strip() == "?":
            self._cmd_help("")
            return True
        return False


def demo_extended():
    """Pattern 3: domain-specific subclass."""
    svc = InventoryService()
    backend = LocalBackend(svc)
    repl = InventoryREPL(backend)

    print("=" * 60)
    print("  Pattern 3: domain-specific REPL")
    print("  InventoryREPL(ServiceREPL) - domain knowledge injected through hooks")
    print("=" * 60)
    print()
    print("  Generic commands (auto-generated):")
    print("    commands / describe / addItem / updateStock / ...")
    print()
    print("  Domain commands (custom):")
    print("    ls [category]       -> compact list")
    print("    low [threshold]     -> low-stock report")
    print("    ?                   -> help (implicit command)")
    print()
    print("  Custom display:")
    print("    addItem -> '✓ Added: ...'  (human-readable instead of JSON)")
    print()

    repl.run()


# =====================================================================
# Main
# =====================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="ServiceREPL Demo")
    parser.add_argument("--extended", action="store_true",
                        help="Run the extended (subclass) demo instead of zero-cost")
    args = parser.parse_args()

    if args.extended:
        demo_extended()
    else:
        demo_zero_cost()


if __name__ == "__main__":
    main()

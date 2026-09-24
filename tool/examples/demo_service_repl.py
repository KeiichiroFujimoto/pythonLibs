#!/usr/bin/env python3.12
"""
ServiceREPL デモ — toolBaseSecured の3つの CLI パターン

このファイル1つで、ServiceREPL の全パターンを理解できます。

=== パターン1: ゼロコスト (svc.toCLI().run()) ===
  開発者が toolBaseSecured サブクラスを書くだけで、CLI が自動生成される。
  追加コード: 0行

=== パターン2: スタンドアロン CLI (URL だけ) ===
  python -m pythonLibs.tool.ServiceREPL http://localhost:8322
  コード不要。サーバ URL を渡すだけ。

=== パターン3: ドメイン特化サブクラス ===
  ServiceREPL を継承して、ドメイン固有の機能を追加。
  EntityNetREPL のように名前解決、独自構文、専用表示を提供。

Usage:
  python demo_service_repl.py              # パターン1 (ゼロコスト)
  python demo_service_repl.py --extended   # パターン3 (ドメイン特化)
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

os.environ["TOOLBASE_SECURED_ENABLED"] = "0"  # デモなので認証無効

from pythonLibs.tool.toolBaseSecured import toolBaseSecured
from pythonLibs.tool.decorators import secure_expose
from pythonLibs.tool.ServiceREPL import ServiceREPL, LocalBackend


# =====================================================================
# Step 1: toolBaseSecured サブクラスを書く (これだけが開発者の仕事)
# =====================================================================
class InventoryService(toolBaseSecured):
    """在庫管理サービスの例。

    @secure_expose を付けたメソッドが自動的に:
      - Web API (buildCatalog → /api/commands)
      - LLM Agent (toLangchainTools → StructuredTool[])
      - CLI/REPL (toCLI → ServiceREPL)
    の3つのインターフェースに展開される。
    """

    
    def __init__(self):
        super().__init__(
            exposure_mode="expose_only",
            secure_enabled=False,
            auth_handler=self._NoOpAuth(),
        )
        # In-memory データストア
        self._items: dict[str, dict] = {
            "item_001": {"name": "Resistor 10kΩ", "category": "electronics", "stock": 500, "unit_price": 0.05},
            "item_002": {"name": "Capacitor 100μF", "category": "electronics", "stock": 200, "unit_price": 0.12},
            "item_003": {"name": "M3 Bolt", "category": "mechanical", "stock": 1000, "unit_price": 0.02},
            "item_004": {"name": "Thermal Paste", "category": "thermal", "stock": 50, "unit_price": 3.50},
            "item_005": {"name": "Heat Sink AL-40", "category": "thermal", "stock": 30, "unit_price": 8.00},
        }
        self._next_id = 6

    # --- 以下、@secure_expose メソッドだけ書けばいい ---

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
# パターン1: ゼロコスト — これだけで CLI 完成
# =====================================================================
def demo_zero_cost():
    """
    svc.toCLI().run() — 追加コード0行で対話的 CLI が使える。

    自動生成されるコマンド:
      commands [category]          → 全オペレーション一覧
      describe <command>           → パラメータ詳細
      <command> key=value ...      → 任意コマンド実行
      help / refresh / quit        → 組み込み
    + Tab 補完 (コマンド名 + パラメータ名)
    """
    svc = InventoryService()

    print("=" * 60)
    print("  パターン1: ゼロコスト CLI")
    print("  svc.toCLI().run() — 追加コード0行")
    print("=" * 60)
    print()
    print("  試してみてください:")
    print("    commands               → カテゴリ別オペレーション一覧")
    print("    describe addItem       → パラメータ名/型/必須/デフォルト")
    print("    listItems              → 全アイテム表示")
    print("    listItems category=thermal  → カテゴリ絞り込み")
    print("    addItem name=\"LED 5mm\" category=electronics stock=100")
    print("    updateStock item_id=item_001 delta=-50")
    print("    stockReport threshold=60")
    print("    totalValue             → 在庫総額")
    print("    searchItems query=heat → 名前検索")
    print()

    # ↓ これだけ。これが全て。
    svc.toCLI(prompt="inventory> ").run()


# =====================================================================
# パターン3: ドメイン特化サブクラス
# =====================================================================
class InventoryREPL(ServiceREPL):
    """在庫管理に特化した REPL。

    ServiceREPL の6つのフックを使ってドメイン知識を注入:
      on_connect()         → 起動時バナー
      custom_commands()    → ドメインコマンド (ls, low)
      extra_completions()  → アイテム名補完
      format_result()      → 結果の見やすい表示
      implicit_command()   → 短縮構文
    """

    def __init__(self, backend):
        super().__init__(backend, prompt="inv> ")
        self._item_names: dict[str, str] = {}  # id → name

    # --- Hook 1: 起動時 ---
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

    # --- Hook 2: ドメインコマンド ---
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

    # --- Hook 3: Tab 補完 ---
    def extra_completions(self, line: str, text: str) -> list[str] | None:
        stripped = line.lstrip()
        parts = stripped.split(None, 1)
        if not parts:
            return None

        cmd = parts[0]

        # ls → カテゴリ補完
        if cmd == "ls":
            cats = set()
            result = self._backend.invoke("listItems", {})
            for item in result.get("items", {}).values():
                cats.add(item.get("category", ""))
            prefix = text.lower()
            return [c for c in sorted(cats) if c.lower().startswith(prefix)]

        return None  # デフォルト補完にフォールスルー

    # --- Hook 4: 結果フォーマット ---
    def format_result(self, command: str, result) -> str | None:
        if command == "addItem":
            return f"  ✓ Added: {result['name']} ({result['item_id']})"
        if command == "updateStock":
            return f"  ✓ Stock: {result['old_stock']} → {result['new_stock']}"
        if command == "deleteItem":
            return f"  ✓ Deleted: {result['name']}"
        if command == "totalValue":
            return f"  Total inventory value: ${result['total_value']:.2f} ({result['item_count']} items)"
        return None  # その他はデフォルト JSON 表示

    # --- Hook 5: 暗黙コマンド ---
    def implicit_command(self, line: str) -> bool:
        # "?" をヘルプのショートカットに
        if line.strip() == "?":
            self._cmd_help("")
            return True
        return False


def demo_extended():
    """パターン3: ドメイン特化サブクラス。"""
    svc = InventoryService()
    backend = LocalBackend(svc)
    repl = InventoryREPL(backend)

    print("=" * 60)
    print("  パターン3: ドメイン特化 REPL")
    print("  InventoryREPL(ServiceREPL) — フックでドメイン知識を注入")
    print("=" * 60)
    print()
    print("  汎用コマンド (自動生成):")
    print("    commands / describe / addItem / updateStock / ...")
    print()
    print("  ドメインコマンド (カスタム):")
    print("    ls [category]       → コンパクト一覧")
    print("    low [threshold]     → 在庫不足レポート")
    print("    ?                   → ヘルプ (暗黙コマンド)")
    print()
    print("  カスタム表示:")
    print("    addItem → '✓ Added: ...'  (JSON ではなく人間向け)")
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

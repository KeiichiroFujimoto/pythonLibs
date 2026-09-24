# Tool Registry Architecture

## 概要
Tool Registryは、ツール（メソッド）の自動検出、登録、検索を管理するためのアーキテクチャです。エージェントが動的にツールを選択・実行できるように設計されています。

## 設計原則

### 1. **Discovery-Driven Design（発見主導設計）**
- モジュールやクラス内の `@secure_expose` デコレータ付きメソッドを自動検出
- 手動登録と自動検出の両方に対応

### 2. **Registry Pattern（レジストリパターン）**
- ツール名をキーに、ツールクラスを値として管理
- ツールの検索・取得を一箇所で管理

### 3. **Name Resolution（名前解決）**
- エージェントが使用する「人間が読みやすい名前」と「実際のメソッド名」のマッピング
- エイリアス機能により、柔軟なツール名を使用可能

## クラス構造

### `ToolRegistry` クラス

**主要な機能:**

| メソッド | 説明 |
|---------|------|
| `register(name, tool_class)` | ツールを登録 |
| `discover_from_module(module)` | モジュール内のツールを自動検出 |
| `get_tool(name)` | ツールクラスを取得 |
| `list_tools(pattern)` | ツール名リストを取得（パターンマッチング対応） |
| `invoke(name, **kwargs)` | ツールをインスタンス化して実行 |

## 使用例

### 基本的な使用方法

```python
from pythonLibs.tool.ToolRegistry import ToolRegistry

# レジストリの作成
registry = ToolRegistry()

# モジュールの自動検出
registry.discover_from_module(your_tool_module)

# ツールの取得
tool_class = registry.get_tool("analyze_data")

# ツールのリスト
available_tools = registry.list_tools("*data*")
# -> ["analyze_data", "process_data", "extract_data"]
```

### カスタム登録

```python
# 既存のクラスを手動で登録
registry.register("custom_tool", CustomToolClass)

# エイリアスの設定
# CustomToolClass に @secure_alias="my_custom_tool" を付ける
# その後、registry.get_tool("my_custom_tool") でも取得可能
```

### パターンマッチング

```python
# 全てのツールを取得
all_tools = registry.list_tools("*")

# 特定の名前パターンにマッチするツールのみ取得
data_tools = registry.list_tools("*data*")
io_tools = registry.list_tools("*io*")
```

## 実装のベストプラクティス

### 1. **モジュールの自動検出**

```python
def discover_from_module(self, module) -> None:
    """@secure_expose デコレータ付きのメソッドを自動検出"""
    for name in dir(module):
        if name.startswith("_"):
            continue
        attr = getattr(module, name)
        if hasattr(attr, "_secure_expose"):
            self.register(name, module)
```

### 2. **名前解決のエラーハンドリング**

```python
def resolve_method_name(self, name: str) -> str:
    actual = self.EXPOSE_ALIASES.get(name, name)
    if not hasattr(self, "_exposed_methods") or actual not in self._exposed_methods:
        raise AttributeError(f"method '{name}' is not exposed")
    return actual
```

### 3. **ツールの実行フロー**

```python
def invoke(self, method_name: str, *args, token: Optional[str] = None, **kwargs) -> Any:
    actual = self.resolve_method_name(method_name)
    if token is not None:
        kwargs.setdefault("token", token)
    method = getattr(self, actual)
    return method(*args, **kwargs)
```

## 拡張性

### カスタム検出ロジック

```python
def discover_from_custom_location(self, location) -> None:
    """カスタムの場所からツールを検出"""
    # 実装例: ファイルシステムからの検出
    # ...
```

### カスタム検索フィルタ

```python
def list_tools_filtered(self, pattern: str, category: str = None) -> List[str]:
    """パターンとカテゴリでフィルタリング"""
    # 実装例: カテゴリ属性に基づいたフィルタリング
    # ...
```

## 関連するクラス

- `toolBaseSecured`: セキュアなツールの基底クラス
- `ToolSecurityManager`: トークン検証とセキュリティ管理
- `secure_expose` デコレータ: メソッドの公開宣言

## 参照

- [Tool Selection Concepts](../concepts/tool-selection.md)

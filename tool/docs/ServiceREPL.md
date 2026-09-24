# ServiceREPL — 使い方ドキュメント

## 概要

`ServiceREPL` は `@secure_expose` デコレータで公開された任意の `executionBaseSecured` サービスに対して、**ゼロ追加コスト**で対話的 CLI を自動生成するフレームワーク。

`toolBaseSecured` は互換名として引き続き使えますが、新規コードでは `executionBaseSecured` を推奨します。

```
@secure_expose  →  5つの出口が自動生成
  ├── Web API        buildCatalog()  → /api/commands
  ├── LLM Agent      toLangchainTools() → StructuredTool[]
  ├── CLI/REPL       toCLI()         → ServiceREPL        ← これ
  ├── GUI            Command Palette (Cmd+K)
  └── Multi-Service  CompositeBackend → 複数サービス統合
```

---

## クイックスタート

### 1. ゼロコスト CLI（コード追加 0 行）

```python
from pythonLibs.tool.executionBaseSecured import executionBaseSecured
from pythonLibs.tool.decorators import secure_expose

class MyService(executionBaseSecured):
    class _NoOpAuth:
        pass

    def __init__(self):
        super().__init__(
            secure_enabled=False,
            auth_handler=self._NoOpAuth(),
            exposure_mode="expose_only",
        )

    @secure_expose(alias="greet", category="social")
    def greet(self, name: str = "World") -> dict:
        """Say hello."""
        return {"greeting": f"Hello, {name}!"}

    @secure_expose(alias="add", category="math")
    def add_numbers(self, x: float, y: float) -> dict:
        """Add two numbers."""
        return {"result": x + y}

# これだけで CLI が使える
svc = MyService()
svc.toCLI().run()
```

実行:
```
$ python my_service.py
  2 commands available. Type 'help' for usage.

service> commands
  [math] (1 commands)
    add                            Add two numbers.  (2/2 params)

  [social] (1 commands)
    greet                          Say hello.  (0/1 params)

service> greet name=Alice
{"greeting": "Hello, Alice!"}

service> add x=3 y=4
{"result": 7.0}

service> quit
Bye.
```

### 2. リモートサーバに接続（スタンドアロン CLI）

```bash
# FiTsZ サーバ（port 8322）に接続
python3.12 -m pythonLibs.tool http://localhost:8322

# プロンプトをカスタマイズ
python3.12 -m pythonLibs.tool http://localhost:8322 --prompt "fitsz> "
```

### 3. 複数サービスを統合

```python
from pythonLibs.tool import ServiceREPL, CompositeBackend

svc_a = ServiceA()
svc_b = ServiceB()

backend = CompositeBackend([svc_a, svc_b])
ServiceREPL(backend, prompt="combined> ").run()
# → 両サービスのコマンドが一つの REPL で使える
```

---

## アーキテクチャ

### Backend Strategy Pattern

```
ServiceBackend (Protocol)
  ├── LocalBackend(service)      直接 Python 呼び出し
  ├── RemoteBackend(url)         HTTP /api/commands
  └── CompositeBackend([svc...]) カタログ合成 + 自動振り分け
```

| Backend | 用途 | 依存 |
|---------|------|------|
| `LocalBackend` | テスト、スクリプト、サーバ不要 | なし |
| `RemoteBackend` | 稼働中サーバに接続 | urllib のみ |
| `CompositeBackend` | 複数サービスを1つの REPL に | なし |

### カタログが中間表現

```python
# buildCatalog() の出力（全インターフェースの共通ソース）
{
    "id": "createEntity",
    "method": "create_entity",
    "description": "Create a new entity.",
    "category": "entity",
    "params": [
        {"name": "name", "type": "string", "required": True, "description": ""},
        {"name": "stereotype", "type": "string", "required": False, "default": "Block", "description": ""}
    ]
}
```

---

## 組み込みコマンド

| コマンド | 説明 |
|----------|------|
| `commands [category]` | コマンド一覧（カテゴリでフィルタ可） |
| `describe <command>` | パラメータ名・型・必須・デフォルト・説明を表示 |
| `<command> key=value ...` | 任意コマンドを実行 |
| `help` | ヘルプ表示 |
| `refresh` | カタログを再取得 |
| `quit` / `exit` | 終了 |

---

## key=value パーサ

### 基本構文

```
command name=value stereotype=Block count=5
```

### 型変換（カタログ駆動）

| カタログ型 | 入力例 | 変換結果 |
|-----------|--------|---------|
| `string` | `name=Sensor` | `"Sensor"` |
| `string` | `name="Heat Exchanger"` | `"Heat Exchanger"` (クォート除去) |
| `integer` | `count=42` | `42` (int) |
| `number` | `ratio=3.14` | `3.14` (float) |
| `boolean` | `flag=true` | `True` |
| `object` | `pos={"x":100,"y":200}` | `{"x": 100, "y": 200}` (dict) |
| `array` | `ids=[1,2,3]` | `[1, 2, 3]` (list) |

### JSON の扱い

`shlex.split()` は JSON のクォートを壊すため、カスタム `_tokenize_kv()` を使用。
ブレース深度を追跡し、`{...}` / `[...]` 内のスペースやクォートを保持する。

---

## Tab 補完

| 入力位置 | 補完候補 |
|----------|---------|
| 行頭 | コマンド名 (builtin + catalog + custom) |
| `commands ` | カテゴリ名 |
| `describe ` | コマンド名 |
| `createEntity ` | パラメータ名 (`name=`, `stereotype=`) |
| (サブクラス) | `extra_completions()` で自由に拡張 |

readline / libedit 両対応（macOS の libedit を自動検出）。

---

## サブクラスで拡張

6つの拡張フック:

```python
class MyREPL(ServiceREPL):

    def on_connect(self) -> str | None:
        """カタログ読み込み後に呼ばれる。バナー文字列を返す。"""
        return f"Connected! {len(self._catalog)} operations."

    def on_refresh(self) -> str | None:
        """refresh 後に呼ばれる。"""
        return None

    def custom_commands(self) -> dict[str, Callable]:
        """ドメイン固有コマンドを登録する。"""
        return {
            "show": self._cmd_show,
            "list": self._cmd_list,
        }

    def extra_completions(self, line: str, text: str) -> list[str] | None:
        """ドメイン固有の Tab 補完。None で汎用にフォールスルー。"""
        if line.startswith("show "):
            return [n for n in self.entity_names if n.startswith(text)]
        return None  # フォールスルー

    def format_result(self, command: str, result: Any) -> str | None:
        """出力フォーマットのカスタマイズ。None でデフォルト JSON。"""
        if command == "getItems":
            return "\n".join(f"  {k}: {v}" for k, v in result["items"].items())
        return None

    def custom_prompt(self) -> str:
        """動的プロンプト。"""
        return f"{self.current_project}> "

    def implicit_command(self, line: str) -> bool:
        """未知入力の処理。True=処理済み、False=エラー表示。"""
        if "=" in line and not line.split()[0] in self._cmd_index:
            # 暗黙の export 構文
            self._handle_export(line)
            return True
        return False
```

---

## API リファレンス

### クラス

| クラス | 説明 |
|--------|------|
| `ServiceREPL(backend, *, prompt=, banner=)` | 汎用 REPL |
| `LocalBackend(service, category_map=)` | 直接呼び出しバックエンド |
| `RemoteBackend(base_url)` | HTTP バックエンド |
| `CompositeBackend(services, category_maps=)` | 複数サービス合成 |

### executionBaseSecured メソッド

| メソッド | 説明 |
|----------|------|
| `svc.buildCatalog(category_map=)` | コマンドカタログ生成 |
| `svc.toCLI(**kwargs)` | `ServiceREPL(LocalBackend(self))` を返す |

### パーサ関数

| 関数 | 説明 |
|------|------|
| `_tokenize_kv(s)` | JSON/クォート対応トークン分割 |
| `_strip_quotes(s)` | 外側の引用符を除去 |
| `_coerce_value(raw, type)` | カタログ型に基づく値変換 |
| `_parse_kv_args(args, specs)` | key=value 文字列を dict に変換 |

### カテゴリ優先順位

`buildCatalog()` のカテゴリ解決:

1. `@secure_expose(category="...")` デコレータ属性 (**最優先**)
2. `category_map` 引数（デコレータなしの場合）
3. `"other"` フォールバック

---

## ファイル構成

```
pythonLibs/tool/
  ├── ServiceREPL.py          メインフレームワーク (~680行)
  ├── executionBaseSecured.py buildCatalog() + toCLI()
  ├── decorators.py           @secure_expose(category=) 追加
  ├── __init__.py             re-export
  ├── __main__.py             python -m pythonLibs.tool URL
  ├── docs/
  │   └── ServiceREPL.md      このドキュメント
  ├── tests/
  │   ├── test_service_repl.py  101 tests
  │   └── RESULTS.md           検証結果
  └── examples/
      └── demo_service_repl.py  InventoryService デモ
```

---

## 実行例

### デモサービス

```bash
python3.12 pythonLibs/tool/examples/demo_service_repl.py
python3.12 pythonLibs/tool/examples/demo_service_repl.py --extended
```

### FiTsZ サーバに接続

```bash
# サーバ起動
python3.12 -m pythonLibs.FiTsZ.backend.server &

# CLI で接続
python3.12 -m pythonLibs.tool http://localhost:8322
```

```
localhost:8322> commands entity
  [entity] (7 commands)
    createEntity                   Create a new entity  (1/4 params)
    deleteEntity                   Delete an entity     (1/1 params)
    ...

localhost:8322> describe createEntity
  createEntity  [entity]
  Create a new entity

  Parameters:
    * name                 string
      stereotype           string   = Block
      parent_id            string
      position             object

localhost:8322> createEntity name="Rocket Engine" stereotype=Block
{"entity_id": "Block_1738...", "name": "Rocket Engine", ...}
```

### テスト実行

```bash
python3.12 -m pytest pythonLibs/tool/tests/test_service_repl.py -v
```

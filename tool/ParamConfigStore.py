# -*- coding: utf-8 -*-
"""ParamConfigStore — generic per-user, per-scope settings persistence.

The make-or-break #4: every settings consumer must NOT reinvent
persistence. This generalises the LLM-specific
``LLMHandler.AgentConfigStore`` (same layered semantics, secret-free,
atomic 0600) to an arbitrary ``scope`` (a service id), so the
SettingsParamsMixin gets durable per-user config for free.

Layered precedence (low → high): defaults (caller) < environment <
per-user file < explicit overrides. One file per user, a ``scopes``
sub-tree keyed by scope (mirrors AgentConfigStore's ``apps`` pattern).
Secrets are never written here (resolved from env / a secret store).

No new package; trunk module in ``pythonLibs/tool``; stdlib only.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1


def config_dir() -> Path:
    d = os.environ.get("PARAM_CONFIG_DIR")
    return Path(d) if d else Path.home() / ".config" / "pulpla" / "params"


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(s)) or "default"


def _user(user_id: Optional[str]) -> str:
    return _safe(user_id or os.environ.get("PARAM_USER_ID")
                 or getpass.getuser())


def config_path(user_id: Optional[str] = None) -> Path:
    return config_dir() / f"{_user(user_id)}.json"


def _load_raw(user_id: Optional[str]) -> Dict[str, Any]:
    p = config_path(user_id)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load(scope: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Stored values for (user, scope). No file ⇒ {}."""
    raw = _load_raw(user_id)
    sc = (raw.get("scopes") or {}).get(scope)
    return dict(sc) if isinstance(sc, dict) else {}


def save(scope: str, values: Dict[str, Any],
         user_id: Optional[str] = None) -> Path:
    """Persist (merge) values under (user, scope). Caller must pass
    secret-free values (the SettingsParamsMixin strips them)."""
    raw = _load_raw(user_id)
    raw["version"] = SCHEMA_VERSION
    scopes = raw.setdefault("scopes", {})
    cur = scopes.get(scope) if isinstance(scopes.get(scope), dict) else {}
    cur.update({k: v for k, v in (values or {}).items() if v is not None})
    scopes[scope] = cur
    path = config_path(user_id)
    _atomic_write(path, json.dumps(raw, ensure_ascii=False, indent=2))
    return path


def resolve(scope: str, *, defaults: Optional[Dict[str, Any]] = None,
            env_prefix: Optional[str] = None,
            user_id: Optional[str] = None,
            overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Layered effective values: defaults < env (``<PREFIX>_<NAME>``,
    upper-case) < per-user file < explicit overrides."""
    merged: Dict[str, Any] = dict(defaults or {})
    if env_prefix:
        pref = env_prefix.upper().rstrip("_") + "_"
        for k in list(merged) + list((overrides or {})):
            ev = os.environ.get(pref + k.upper())
            if ev is not None and ev != "":
                merged[k] = ev
    merged.update(load(scope, user_id))
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


__all__ = ["SCHEMA_VERSION", "config_dir", "config_path",
           "load", "save", "resolve"]

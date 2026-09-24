# -*- coding: utf-8 -*-
"""SettingsParams — turn any toolBaseSecured's parameters into a common,
auto-generated settings UI with near-zero per-app code.

Model (user's design):
  * the service sets the parameters it handles into ``paramDict``;
  * it declares ``USER_PARAMS`` = the subset that needs user input
    (optionally ``PARAM_META`` for label/type/options/secret/…);
  * ``paramsSpec()`` projects (paramDict ∩ USER_PARAMS) into a
    GenerativeUI-consumable field list → the webCore SettingsPane
    renders it via runtime-generative-ui; ``setParams`` writes back.
  * ``serializeParams`` emits the same values as JSON / XML / TOML —
    formats are *projections*, the paramDict is the single source.

Make-or-break points designed in from the start:
  #1 dependent/dynamic options — a field may declare ``optionsFrom``
     (another @secure_expose command) + ``dependsOn`` so the UI can
     refetch options when a parent field changes (the provider→model
     case). The descriptor carries it; the Pane resolves it live.
  #2 secrets — explicit ``secret: True`` or a name hint → never echoed
     back in plaintext, excluded from serialize by default.
  #3 a dedicated paramDict sub-namespace (``settings``) so user config
     never collides with execution scratch (executionContext / result).
  (#4 layered per-user persistence is the sibling ParamConfigStore.)

Opt-in mixin (a service adds it explicitly — non-destructive, no change
to the hot toolBase/toolBaseSecured cores). No new package.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
from xml.sax.saxutils import escape as _xml_escape

from pythonLibs.tool.decorators import secure_expose

SETTINGS_NS = "settings"

# Secondary heuristic only — explicit PARAM_META {"secret": True} wins.
_SECRET_HINT = ("api_key", "apikey", "secret", "token", "password",
                "passwd", "authorization", "credential", "private_key")


def _infer_type(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    return "string"


def _to_xml(root: str, d: Dict[str, Any]) -> str:
    parts = [f"<{root}>"]
    for k, v in d.items():
        parts.append(f"  <{k}>{_xml_escape(str(v))}</{k}>")
    parts.append(f"</{root}>")
    return "\n".join(parts)


def _from_xml(text: str) -> Dict[str, Any]:
    root = ET.fromstring(text)
    return {child.tag: (child.text or "") for child in root}


class SettingsParamsMixin:
    """Mix into a ``toolBaseSecured``. Declare ``USER_PARAMS`` (and
    optionally ``PARAM_META``); the settings surface is then free."""

    USER_PARAMS: List[str] = []
    PARAM_META: Dict[str, Dict[str, Any]] = {}
    SETTINGS_TITLE: str = ""
    SETTINGS_SCOPE: str = ""          # ParamConfigStore key (default: class)
    SETTINGS_ENV_PREFIX: str = ""     # optional env overlay prefix

    # ── paramDict sub-namespace (isolated from exec scratch) ───────────
    def _settings_store(self) -> Dict[str, Any]:
        pd = self.paramDict
        cur = pd.get(SETTINGS_NS) if hasattr(pd, "get") else None
        if not isinstance(cur, dict):
            pd[SETTINGS_NS] = {}
        s = pd[SETTINGS_NS]
        for n in self.USER_PARAMS:                       # seed defaults
            if n not in s:
                meta = self.PARAM_META.get(n, {})
                if "default" in meta:
                    s[n] = meta["default"]
        return s

    def _is_secret(self, name: str) -> bool:
        if self.PARAM_META.get(name, {}).get("secret"):
            return True
        ln = name.lower()
        return any(h in ln for h in _SECRET_HINT)

    def _coerce(self, name: str, value: Any) -> Any:
        t = self.PARAM_META.get(name, {}).get("type") or ""
        try:
            if t == "int":
                return int(value)
            if t in ("float", "number"):
                return float(value)
            if t == "bool":
                return (value if isinstance(value, bool)
                        else str(value).strip().lower()
                        in ("1", "true", "yes", "on"))
        except (TypeError, ValueError):
            return value
        return value

    # ── the common surface (1 def → 6 interfaces via toolBase) ─────────
    @secure_expose(alias="paramsSpec", category="settings")
    def paramsSpec(self) -> Dict[str, Any]:
        """GenerativeUI-consumable field list for (paramDict ∩
        USER_PARAMS). Secret values are withheld; dynamic options carry
        ``optionsFrom``/``dependsOn`` so the Pane resolves them live."""
        s = self._settings_store()
        fields: List[Dict[str, Any]] = []
        for n in self.USER_PARAMS:
            m = self.PARAM_META.get(n, {})
            secret = self._is_secret(n)
            f: Dict[str, Any] = {
                "name": n,
                "label": m.get("label", n),
                "type": m.get("type") or _infer_type(s.get(n)),
                "group": m.get("group", "General"),
                "secret": secret,
                "value": None if secret else s.get(n),
            }
            if "options" in m:
                f["options"] = m["options"]
            if "optionsFrom" in m:                       # make-or-break #1
                f["optionsFrom"] = m["optionsFrom"]
                if "dependsOn" in m:
                    f["dependsOn"] = m["dependsOn"]
            for opt in ("placeholder", "min", "max", "help", "required"):
                if opt in m:
                    f[opt] = m[opt]
            fields.append(f)
        return {
            "title": self.SETTINGS_TITLE or type(self).__name__,
            "fields": fields,
            "secretKeys": [n for n in self.USER_PARAMS
                           if self._is_secret(n)],
        }

    @secure_expose(alias="getParams", category="settings")
    def getParams(self) -> Dict[str, Any]:
        """Current values to pre-fill the form. Secret fields return a
        presence flag, never the value."""
        s = self._settings_store()
        out: Dict[str, Any] = {}
        for n in self.USER_PARAMS:
            if self._is_secret(n):
                out[n] = {"__secret__": True, "set": bool(s.get(n))}
            else:
                out[n] = s.get(n)
        return {"params": out}

    @secure_expose(alias="setParams", category="settings")
    def setParams(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Write user-supplied values (only USER_PARAMS, type-coerced)
        into the settings namespace. Secrets are stored but echoed
        masked."""
        s = self._settings_store()
        allowed = set(self.USER_PARAMS)
        applied: Dict[str, Any] = {}
        for k, v in (values or {}).items():
            if k not in allowed or v is None:
                continue
            s[k] = self._coerce(k, v)
            applied[k] = "***" if self._is_secret(k) else s[k]
        return {"applied": applied, "rejected":
                sorted(set((values or {})) - allowed)}

    @secure_expose(alias="serializeParams", category="settings")
    def serializeParams(self, format: str = "json",
                        includeSecrets: bool = False) -> Dict[str, Any]:
        """Project the settings as JSON / TOML / XML (a derived view —
        secrets excluded unless explicitly requested)."""
        s = dict(self._settings_store())
        if not includeSecrets:
            for n in list(s):
                if self._is_secret(n):
                    s.pop(n, None)
        fmt = (format or "json").lower()
        try:
            if fmt == "json":
                text = json.dumps(s, ensure_ascii=False, indent=2)
            elif fmt == "toml":
                import toml
                text = toml.dumps(s)
            elif fmt == "xml":
                text = _to_xml(SETTINGS_NS, s)
            else:
                return {"error": f"unknown format: {format}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"serialize failed: {exc}"}
        return {"format": fmt, "text": text}

    # ── make-or-break #4: durable per-user persistence (generalised,
    #    reused — not reinvented per service) ────────────────────────
    def _scope(self) -> str:
        return self.SETTINGS_SCOPE or type(self).__name__

    @secure_expose(alias="saveParams", category="settings")
    def saveParams(self, userId: str = None) -> Dict[str, Any]:
        """Persist the user's non-secret settings to the layered
        per-user store. Secrets stay out of the file (env/secret-store)."""
        from pythonLibs.tool import ParamConfigStore as _store
        s = self._settings_store()
        payload = {n: s.get(n) for n in self.USER_PARAMS
                   if not self._is_secret(n) and s.get(n) is not None}
        path = _store.save(self._scope(), payload, user_id=userId)
        return {"saved": True, "scope": self._scope(),
                "path": str(path), "keys": sorted(payload)}

    @secure_expose(alias="loadUserParams", category="settings")
    def loadUserParams(self, userId: str = None) -> Dict[str, Any]:
        """Hydrate the settings namespace from the layered store
        (defaults < env < per-user file). Secrets untouched."""
        from pythonLibs.tool import ParamConfigStore as _store
        defaults = {n: self.PARAM_META.get(n, {}).get("default")
                    for n in self.USER_PARAMS
                    if "default" in self.PARAM_META.get(n, {})}
        eff = _store.resolve(
            self._scope(), defaults=defaults,
            env_prefix=self.SETTINGS_ENV_PREFIX or None, user_id=userId,
        )
        return self.setParams({k: v for k, v in eff.items()
                               if k in set(self.USER_PARAMS)
                               and not self._is_secret(k)})

    @secure_expose(alias="loadParams", category="settings")
    def loadParams(self, text: str,
                   format: str = "json") -> Dict[str, Any]:
        """Inverse of serializeParams (round-trips JSON/TOML/XML)."""
        fmt = (format or "json").lower()
        try:
            if fmt == "json":
                data = json.loads(text)
            elif fmt == "toml":
                import tomllib
                data = tomllib.loads(text)
            elif fmt == "xml":
                data = _from_xml(text)
            else:
                return {"error": f"unknown format: {format}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"parse failed: {exc}"}
        if not isinstance(data, dict):
            return {"error": "expected a mapping"}
        return self.setParams({k: v for k, v in data.items()
                               if k in set(self.USER_PARAMS)})


__all__ = ["SettingsParamsMixin", "SETTINGS_NS"]

"""
Cross-platform helpers for reading environment variables from different scopes.

This module centralises OS-specific fallbacks (such as Windows machine-level
environment values) so higher level libraries can remain platform agnostic.
"""

from __future__ import annotations

import os
import platform
from typing import Optional


def get_process_env_var(name: str) -> Optional[str]:
    """Return environment variable from the current process."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_windows_machine_env_var(name: str) -> Optional[str]:
    """
    Retrieve machine-level environment variables on Windows.

    On non-Windows platforms this always returns None.
    """
    if platform.system().lower() != "windows":
        return None
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ) as key:
            value, _ = winreg.QueryValueEx(key, name)
            if isinstance(value, str):
                value = value.strip()
                return value or None
    except (FileNotFoundError, OSError, ImportError):
        return None
    return None


def resolve_env_var(name: str) -> Optional[str]:
    """
    Resolve an environment variable considering multiple scopes.

    Priority:
    1. Process environment (works on all platforms)
    2. Windows machine-level environment variables
    """
    value = get_process_env_var(name)
    if value:
        return value
    return get_windows_machine_env_var(name)


__all__ = ["get_process_env_var", "get_windows_machine_env_var", "resolve_env_var"]

from __future__ import annotations
from typing import Optional

class SecurityVerificationError(PermissionError):
    """Raised when an injected verifier rejects a request."""

class ToolSecurityManager:
    """
    Thin adapter around an application-provided verifier.

    The public-domain toolBase package intentionally does not ship login,
    password, JWT, database, or web-auth implementations. Applications that
    need access control should inject their own handler object. Supported
    handler shapes are deliberately small:

    * ``verify(token) -> bool`` or ``verify_token(token) -> bool``
    * optional ``get_username(token)`` or ``getUsernameByToken(token)``
    """

    def __init__(self, handler: Optional[object] = None, secure_enabled: bool = True):
        self.secure_enabled = bool(secure_enabled)
        self._handler = handler
        self._access_token: Optional[str] = None
        self._username: Optional[str] = None

    def set_token(self, token: Optional[str]) -> None:
        self._access_token = token

    def get_token(self) -> Optional[str]:
        return self._access_token

    def set_username(self, username: Optional[str]) -> None:
        self._username = username

    def get_username(self) -> Optional[str]:
        return self._username

    def _fetch_username_from_handler(self, token: Optional[str]) -> Optional[str]:
        if self._handler is None:
            return None
        try:
            for name in ("get_username", "getUsernameByToken"):
                getter = getattr(self._handler, name, None)
                if callable(getter):
                    return getter(token or self._access_token)
        except Exception:
            return None
        return None

    def refresh_username(self, token: Optional[str] = None) -> Optional[str]:
        self._username = self._fetch_username_from_handler(token)
        return self._username

    def enable(self) -> None:
        self.secure_enabled = True

    def disable(self) -> None:
        self.secure_enabled = False

    def verify_or_raise(self, token: Optional[str] = None) -> None:
        """Verify the token or raise SecurityVerificationError when invalid/missing.
        On success, cache the username automatically.
        """
        if not self.secure_enabled:
            return
        tok = token or self._access_token
        if not tok:
            raise SecurityVerificationError("access token is required")
        if not self._handler:
            raise SecurityVerificationError("verification handler is not initialized")
        verifier = None
        for name in ("verify", "verify_token"):
            candidate = getattr(self._handler, name, None)
            if callable(candidate):
                verifier = candidate
                break
        if verifier is None:
            raise SecurityVerificationError("verification handler has no verify method")
        try:
            accepted = verifier(tok)
        except Exception as exc:
            raise SecurityVerificationError("verification failed") from exc
        if not accepted:
            raise SecurityVerificationError("verification failed")
        self._username = self._fetch_username_from_handler(tok)

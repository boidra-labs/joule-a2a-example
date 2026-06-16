"""XSUAA JWT validation middleware — reads credentials from VCAP_SERVICES xsuaa binding."""
from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from typing import Callable

# Populated by XSUAAAuthMiddleware; readable anywhere in the same async context.
current_user_id: ContextVar[str] = ContextVar("current_user_id", default="")

import jwt
from jwt import PyJWKClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


def _xsuaa_creds() -> dict:
    vcap = json.loads(os.environ.get("VCAP_SERVICES", "{}"))
    bindings = vcap.get("xsuaa", [])
    if not bindings:
        raise RuntimeError("No xsuaa binding found in VCAP_SERVICES")
    return bindings[0]["credentials"]


class XSUAAAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        try:
            creds = _xsuaa_creds()
            self._audience = creds["xsappname"]
            self._jwks_uri = creds.get("jwks_uri") or (
                creds["url"].rstrip("/") + "/token_keys"
            )
            self._jwks_client = PyJWKClient(self._jwks_uri)
            self._enabled = True
        except RuntimeError:
            logger.warning("No XSUAA binding found — JWT validation disabled (local dev mode)")
            self._enabled = False

    async def dispatch(self, request: Request, call_next: Callable):
        # Health check and agent discovery endpoints must be public
        if request.url.path in ("/.well-known/agent-card.json", "/health"):
            return await call_next(request)

        # Local dev: no XSUAA binding → pass through all requests
        if not self._enabled:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            logger.warning("JWT rejected: missing or malformed Authorization header")
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        try:
            token = auth[7:]
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            decoded = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                options={"verify_aud": False},  # XSUAA aud can be a list or string
            )
            # Manual audience check — XSUAA tokens carry both "sb-<xsappname>" and "<xsappname>"
            aud = decoded.get("aud", [])
            if isinstance(aud, str):
                aud = [aud]
            allowed = {self._audience, f"sb-{self._audience}"}
            if not allowed.intersection(aud):
                logger.warning("JWT audience mismatch: token has %s, expected one of %s", aud, allowed)
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

            user_id = decoded.get("user_name") or decoded.get("sub", "unknown")
            request.state.user_id = user_id
            request.state.tenant_id = decoded.get("zid", "")
            current_user_id.set(user_id)
        except Exception as exc:
            logger.warning("JWT validation failed: %s", exc)
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

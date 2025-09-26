"""
API package for the LLM Chat Backend.

Exposes the FastAPI application instance for ASGI servers.
"""

from .main import app  # re-export for convenience

__all__ = ["app"]

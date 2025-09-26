from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from starlette.responses import JSONResponse
from starlette.requests import ClientDisconnect

# PUBLIC_INTERFACE
def get_env(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with an optional default."""
    return os.getenv(var_name, default)


# Basic configuration via environment variables (do not hardcode secrets)
APP_NAME = "LLM Chat Backend"
APP_VERSION = "1.0.0"
APP_DESC = (
    "Modern, minimalist REST API for chat completion using OpenAI-compatible APIs. "
    "Theme: Ocean Professional (blue & amber accents)."
)

# Environment variables required:
# - OPENAI_API_KEY: API key for OpenAI
# Optional:
# - OPENAI_API_BASE: Custom base URL for OpenAI-compatible APIs
# - OPENAI_MODEL: Default model to use (e.g., 'gpt-4o-mini' or 'gpt-4o')
# - ALLOW_ORIGINS: Comma-separated list of CORS origins (default '*')


# Pydantic models for requests and responses
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="The role of the message author."
    )
    content: str = Field(
        ...,
        description="Content of the message."
    )


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        ...,
        description="Ordered list of prior conversation messages ending with the user message."
    )
    model: Optional[str] = Field(
        default=None,
        description="Model ID to use. Overrides server default when provided."
    )
    temperature: Optional[float] = Field(
        default=0.7,
        description="Sampling temperature to use, between 0 and 2."
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate in the chat completion."
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling probability."
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to request server-side streaming. Currently not implemented; must be false."
    )


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = Field(..., description="Role of the assistant response.")
    content: str = Field(..., description="Assistant message content.")


class Choice(BaseModel):
    index: int = Field(..., description="Index of the choice in the result set.")
    message: ChoiceMessage = Field(..., description="The assistant message for this choice.")
    finish_reason: Optional[str] = Field(
        None,
        description="Reason the model stopped generating tokens."
    )


class Usage(BaseModel):
    prompt_tokens: Optional[int] = Field(None, description="Count of tokens in the prompt.")
    completion_tokens: Optional[int] = Field(None, description="Count of tokens in the completion.")
    total_tokens: Optional[int] = Field(None, description="Total tokens used.")


class ChatResponse(BaseModel):
    id: Optional[str] = Field(None, description="Identifier for the response.")
    object: str = Field("chat.completion", description="Object type.")
    created: int = Field(..., description="Unix timestamp of creation.")
    model: str = Field(..., description="Model used for the completion.")
    choices: List[Choice] = Field(..., description="List of response choices.")
    usage: Optional[Usage] = Field(None, description="Token usage information.")


class ServerInfo(BaseModel):
    name: str = Field(..., description="Application name.")
    version: str = Field(..., description="Application semantic version.")
    theme: Dict[str, str] = Field(..., description="Theme color tokens for UI clients.")
    openai_configured: bool = Field(..., description="Whether an OpenAI API key is configured.")
    default_model: Optional[str] = Field(None, description="Default model if configured.")
    api_base: Optional[str] = Field(None, description="OpenAI API base URL in use.")


# Security dependency for OpenAI API key
# PUBLIC_INTERFACE
def require_openai_key(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> str:
    """Validate and return the OpenAI API key from Authorization header or environment.

    Auth precedence:
    1) Bearer <key> in the Authorization header
    2) OPENAI_API_KEY environment variable

    Raises:
        HTTPException: 401 if no key is provided.
    """
    # Try header "Authorization: Bearer sk-..."
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]

    # Fallback to env OPENAI_API_KEY
    env_key = get_env("OPENAI_API_KEY", None)
    if env_key:
        return env_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="OpenAI API key is required. Provide via Authorization: Bearer <key> or set OPENAI_API_KEY.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_app() -> FastAPI:
    """Factory to create and configure the FastAPI app with routes and middleware."""
    app = FastAPI(
        title=APP_NAME,
        description=APP_DESC,
        version=APP_VERSION,
        contact={
            "name": "LLM Chat Backend",
            "url": "https://example.com",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=[
            {"name": "Health", "description": "Service health and metadata."},
            {"name": "Chat", "description": "Chat completion endpoints powered by OpenAI."},
            {"name": "Docs", "description": "Documentation and usage notes."},
        ],
        swagger_ui_parameters={
            # Ocean Professional theme influence: light scheme with primary blue accents.
            "docExpansion": "list",
            "defaultModelsExpandDepth": 0,
            "syntaxHighlight.theme": "agate",
            "filter": True,
            "displayRequestDuration": True,
        },
    )

    # CORS
    allow_origins_env = get_env("ALLOW_ORIGINS", "*")
    allow_origins = [o.strip() for o in allow_origins_env.split(",")] if allow_origins_env else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
        max_age=3600,
    )

    # Exception handlers
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "message": "Validation error."},
        )

    @app.exception_handler(ClientDisconnect)
    async def client_disconnect_handler(request: Request, exc: ClientDisconnect) -> Response:
        return Response(status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST)

    # Routes

    # PUBLIC_INTERFACE
    @app.get(
        "/",
        summary="Health Check",
        description="Basic health check endpoint indicating that the service is running.",
        tags=["Health"],
        responses={
            200: {"description": "Healthy"},
        },
    )
    async def health_check() -> Dict[str, str]:
        """Health check endpoint returning a simple message."""
        return {"message": "Healthy"}

    # PUBLIC_INTERFACE
    @app.get(
        "/info",
        summary="Server Info",
        description="Returns server metadata, theme colors, and OpenAI configuration presence.",
        tags=["Health"],
        response_model=ServerInfo,
    )
    async def info() -> ServerInfo:
        """Return server configuration info without exposing secrets."""
        theme = {
            "name": "Ocean Professional",
            "primary": "#2563EB",
            "secondary": "#F59E0B",
            "success": "#F59E0B",
            "error": "#EF4444",
            "background": "#f9fafb",
            "surface": "#ffffff",
            "text": "#111827",
            "gradient": "from-blue-500/10 to-gray-50",
        }
        api_key_set = get_env("OPENAI_API_KEY") is not None
        default_model = get_env("OPENAI_MODEL", "gpt-4o-mini")
        api_base = get_env("OPENAI_API_BASE")
        return ServerInfo(
            name=APP_NAME,
            version=APP_VERSION,
            theme=theme,
            openai_configured=api_key_set,
            default_model=default_model,
            api_base=api_base,
        )

    # PUBLIC_INTERFACE
    @app.get(
        "/docs/websocket",
        summary="WebSocket Usage",
        description=(
            "This service currently exposes only HTTP REST endpoints for chat completion. "
            "No WebSocket endpoints are available at this time."
        ),
        tags=["Docs"],
        responses={200: {"description": "Usage info."}},
    )
    async def websocket_docs() -> Dict[str, Any]:
        """Provide websocket usage information note."""
        return {
            "websocket_available": False,
            "message": "No WebSocket endpoints available. Use POST /v1/chat/completions for chat.",
        }

    # PUBLIC_INTERFACE
    @app.post(
        "/v1/chat/completions",
        summary="Create chat completion",
        description=(
            "Submit a list of messages and receive an assistant response. "
            "This endpoint is compatible with OpenAI-style chat completion requests."
        ),
        tags=["Chat"],
        response_model=ChatResponse,
        responses={
            200: {"description": "Successful chat completion."},
            400: {"description": "Invalid request."},
            401: {"description": "Unauthorized - missing/invalid API key."},
            502: {"description": "Upstream error from OpenAI."},
        },
    )
    async def chat_completions(
        payload: ChatRequest,
        request: Request,
        api_key: str = Depends(require_openai_key),
    ) -> ChatResponse:
        """Create a chat completion using OpenAI.

        Parameters:
        - payload: ChatRequest body including messages and optional params
        - Authorization: Bearer <OPENAI_API_KEY> header or OPENAI_API_KEY environment variable

        Returns:
        - ChatResponse containing assistant message and metadata
        """
        if payload.stream:
            # For simplicity, we do not implement server-side streaming in this version.
            raise HTTPException(status_code=400, detail="Streaming not supported in this endpoint.")

        # Resolve model and API base
        model = payload.model or get_env("OPENAI_MODEL", "gpt-4o-mini")
        api_base = get_env("OPENAI_API_BASE", "https://api.openai.com/v1")

        # Build request to OpenAI chat completions
        # We intentionally use httpx directly to avoid adding heavy SDKs.
        import httpx

        # Convert messages to dicts
        messages = [m.model_dump() for m in payload.messages]

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if payload.temperature is not None:
            body["temperature"] = payload.temperature
        if payload.max_tokens is not None:
            body["max_tokens"] = payload.max_tokens
        if payload.top_p is not None:
            body["top_p"] = payload.top_p

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = f"{api_base.rstrip('/')}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                upstream = await client.post(url, headers=headers, json=body)
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Upstream timeout from OpenAI.")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"HTTP error contacting OpenAI: {str(e)}")

        if upstream.status_code >= 400:
            # Try to pass upstream error details
            try:
                err_json = upstream.json()
            except Exception:
                err_json = {"message": upstream.text}
            raise HTTPException(
                status_code=502,
                detail={"upstream_status": upstream.status_code, "error": err_json},
            )

        data = upstream.json()

        # Normalize to our ChatResponse schema (compatible with OpenAI response)
        created_ts = data.get("created", int(time.time()))
        response_model = data.get("model", model)

        # Extract the first choice content safely
        choices_out: List[Choice] = []
        for idx, c in enumerate(data.get("choices", [])):
            msg = c.get("message") or {}
            content = msg.get("content", "")
            finish_reason = c.get("finish_reason")
            choices_out.append(
                Choice(
                    index=idx,
                    message=ChoiceMessage(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            )

        usage_in = data.get("usage") or {}
        usage_obj = Usage(
            prompt_tokens=usage_in.get("prompt_tokens"),
            completion_tokens=usage_in.get("completion_tokens"),
            total_tokens=usage_in.get("total_tokens"),
        )

        return ChatResponse(
            id=data.get("id"),
            object=data.get("object", "chat.completion"),
            created=created_ts,
            model=response_model,
            choices=choices_out or [
                Choice(index=0, message=ChoiceMessage(role="assistant", content=""), finish_reason=None)
            ],
            usage=usage_obj,
        )

    return app


# App instance for ASGI servers
app = create_app()

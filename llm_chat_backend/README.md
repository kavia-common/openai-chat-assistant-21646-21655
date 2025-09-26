# LLM Chat Backend (FastAPI)

A modern, minimalist REST API for chat completion using OpenAI-compatible APIs.

Theme: Ocean Professional — Blue primary (#2563EB), amber accents (#F59E0B), subtle gradients and clean layout reflected in API docs metadata.

## Features

- FastAPI-based REST endpoints
- Health and server info endpoints
- Chat completion endpoint compatible with OpenAI `/v1/chat/completions`
- API key authentication via Authorization: Bearer or environment variable
- CORS configurable via environment
- Robust error handling and clear OpenAPI/Swagger docs

## Endpoints

- GET `/` — Health Check
- GET `/info` — Server info and theme tokens (no secrets)
- POST `/v1/chat/completions` — Submit chat messages, returns assistant response
- GET `/docs/websocket` — WebSocket usage note (no websockets in this version)

## Request example

```
POST /v1/chat/completions
Authorization: Bearer sk-...
Content-Type: application/json

{
  "model": "gpt-4o-mini",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello! Provide 3 bullet points why the ocean is important." }
  ],
  "temperature": 0.7
}
```

## Response example (truncated)

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1738000000,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 85,
    "total_tokens": 110
  }
}
```

## Environment Variables

Create a `.env` file in the service directory or set the variables in your runtime environment.

Required:
- `OPENAI_API_KEY` — API key for OpenAI

Optional:
- `OPENAI_API_BASE` — Override API base (default: `https://api.openai.com/v1`)
- `OPENAI_MODEL` — Default model (default: `gpt-4o-mini`)
- `ALLOW_ORIGINS` — Comma-separated CORS origins (default: `*`)

Note: Do not commit actual secret values to version control. Use a `.env` file locally and managed secrets in production.

Example `.env.example`:
```
OPENAI_API_KEY=sk-REPLACE_ME
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
ALLOW_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Running locally

Install dependencies (Python 3.10+ recommended):

```
pip install -r requirements.txt
```

Start the server:

```
uvicorn src.api.main:app --host 0.0.0.0 --port 3001 --proxy-headers
```

OpenAPI docs:
- Swagger UI: `http://localhost:3001/docs`
- OpenAPI JSON: `http://localhost:3001/openapi.json`

## Security

- Provide the API key either via `Authorization: Bearer <key>` or `OPENAI_API_KEY` environment variable.
- The `/info` endpoint exposes only non-sensitive configuration details and style tokens.
- CORS is configurable; restrict `ALLOW_ORIGINS` in production.
- Timeouts and upstream error handling are implemented for reliability.

## License

MIT

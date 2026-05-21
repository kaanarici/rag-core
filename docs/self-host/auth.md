# Self-host auth (app-owned)

`rag-core serve` is a **thin retrieval runtime**. It does not ship accounts, API keys, tenancy, or billing. Your application (or API gateway) owns authentication and authorization.

## Design rule

- **Embed path:** your app constructs `RAGCore` with credentials and enforces auth before calling search/ingest.
- **Serve path:** wrap `serve` behind the same auth middleware you would use for any internal microservice.

Never expose `serve` on the public internet without auth in front of it.

## Recommended pattern

1. Terminate TLS at your gateway (nginx, Envoy, cloud load balancer).
2. Validate identity (JWT, session cookie, mTLS, or API key) **before** proxying to `rag-core serve`.
3. Map the authenticated principal to `namespace` / `corpus_id` in request bodies — do not let clients pick arbitrary tenancy without checks.

## Starlette middleware example (API key)

Add middleware in your deployment wrapper or fork of `create_app` (v1 does not enable auth inside the library):

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from rag_core.runtime.errors import api_error


class ApiKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, expected_key: str, header: str = "x-api-key") -> None:
        super().__init__(app)
        self._expected_key = expected_key
        self._header = header.lower().encode()

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/health"):
            return await call_next(request)
        provided = request.headers.get(self._header.decode())
        if provided != self._expected_key:
            return api_error(
                code="unauthorized",
                message="Invalid or missing API key",
                status_code=401,
            )
        return await call_next(request)
```

Wire it when building the app:

```python
from starlette.middleware import Middleware

app = create_app(...)
app.add_middleware(ApiKeyMiddleware, expected_key=os.environ["RAG_CORE_API_KEY"])
```

Health endpoints (`/health`, `/health/ready`) are often left unauthenticated for orchestrators; lock them down if your platform requires it.

## Tenancy binding

HTTP bodies accept `namespace` and `corpus_id` / `corpus_ids`. In production:

- Derive `namespace` from the authenticated tenant (server-side), not from an unvalidated client field.
- Reject cross-tenant `corpus_ids` in your middleware or a thin BFF layer.

The library does not interpret JWT claims — that stays in your app.

## What rag-core will not add in v1

- User tables, OAuth flows, or API key storage
- Per-route RBAC inside `serve`
- Connector credential vaults

See [quickstart.md](quickstart.md) for running `serve`, [openapi.yaml](openapi.yaml) for the HTTP contract, and [expectations.md](../expectations.md) for retrieval shapes.

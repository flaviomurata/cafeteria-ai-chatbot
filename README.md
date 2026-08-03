# Café Aurora chatbot

This repository is a monorepo for the Café Aurora partner-knowledge chatbot.

## Layout

```
apps/
  api/   FastAPI application, tests, and partner documents
  web/   React browser interface and its Nginx container
compose.yaml
justfile
```

The browser reaches the API through `/api`. The web container proxies that path to
the private `agent-api` container, so the browser never needs to know a Docker host
name or a separate API URL.

## Run the full stack

1. Copy `.env.example` to `.env` and set the provider credentials.
2. Build the knowledge index with `just ingest`.
3. Start both containers with `just up`.
4. Open `http://localhost:3000`.

Useful commands:

- `just api` runs the FastAPI app with reload on port 8000.
- `just test` runs the API test suite.
- `just lint` formats and checks the API code.
- `just ps` shows the container status.

# Grounded Response Contract evaluation

The checked-in matrix at `tests/fixtures/grounded_response_cases.json` is the
release contract for the Partner-facing API. It covers supported answers,
citations, scope refusals, compound unsupported claims, conflicts, direct and
indirect prompt injection, cache behavior, and grounding dependency failures.

## Local deterministic E2E

Run the black-box Compose suite with deterministic local adapters:

```shell
just e2e-local
```

This suite is safe for CI because it does not call an embedding, generation, or
verification provider. It must pass before a release proceeds.

## Live staging evaluation

The live suite is opt-in and refuses to run unless the target is explicitly
identified as staging:

```shell
E2E_BASE_URL=https://staging.example.test just e2e-live
```

The staging deployment must expose `/health` with `environment: staging` and
must use the pinned immutable Partner knowledge index, retrieval configuration,
prompts, and model configuration. A live run fails if the service is missing,
unhealthy, or reports another environment. Grounded cases use a zero-tolerance
policy for unsupported factual claims; missed supported answers are reported
separately by the evaluation process.

## Immutable knowledge index

The API mounts the index read-only and writes query caches and locks to the
separate runtime volume. Ingestion is an explicit operation and is not part of
API startup. Rebuilding the index is a controlled release action because it
calls the configured embedding provider and changes the persistent evidence
set; the rebuilt index must contain exactly the six approved source documents.

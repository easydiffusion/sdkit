# API conventions

sdkit v3 exposes an HTTP API that follows the conventions of the Automatic1111 / Forge
webui. All generation and options endpoints live under one base URL:

```
http://127.0.0.1:<port>/v1/sdapi/v1/
```

A few sdkit-specific endpoints (ping, progress) live under `/v1/internal/`.

The server binds to `127.0.0.1` only, so it is reachable from the same machine. The
default port is `8188` (see [`--port`](../server.md#ports-and-logging)).

## Request and response bodies

All endpoints accept and return JSON (`Content-Type: application/json`):

- Requests are JSON objects; unknown fields are ignored, missing fields fall back to
  defaults.
- Responses are JSON objects on success, or an error object on failure.

## Images

Images are always passed and returned as **base64-encoded strings** - no multipart
forms, no URLs. A request field that takes one image is a base64 string (e.g. the
img2img `mask`); fields that take several are arrays of base64 strings (e.g.
`init_images`, `ref_images`). Responses return arrays of base64 PNG strings.

## Errors

Failures return a non-2xx HTTP status with a JSON body containing a `message` field:

```json
{ "message": "Generation failed: <details>" }
```

| Status | Meaning |
|---|---|
| `400` | Malformed request (e.g. invalid JSON body, missing required field) |
| `404` | Unknown resource (e.g. unknown task ID on the progress endpoint) |
| `500` | Server-side failure (e.g. model not found, upscaler not configured) |

See the [troubleshooting page](../appendix.md) for common errors and fixes.

## Task IDs

Every generation request runs as a *task*. Requests may carry an optional
`force_task_id` field; it defaults to `"default_task"`. The task ID is what you use to
[poll progress](generate.md#progress-polling) - pass it as `id_task` to the
progress endpoint.

!!! note

    The server tracks progress per task ID. If you issue overlapping generations,
    give each a distinct `force_task_id` so their progress can be told apart.

## Blocking behavior

Generation endpoints (`txt2img`, `img2img`) **block until the image is fully
generated**, then return the complete result in the response. Clients that only care
about the final image can just wait for the HTTP response.

Clients that want progress reporting or live previews should:

1. Send the generation request with a `force_task_id`.
2. Poll `POST /v1/internal/progress` with `{"id_task": "<task id>"}` while the
   generation request is in flight.

See [Image generation](generate.md#progress-polling) for the polling loop
and the [interrupt endpoint](generate.md#interrupt).
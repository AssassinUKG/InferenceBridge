# Debug API Workspace

InferenceBridge now treats the embedded API as a first-class part of the app, not a sidecar afterthought. The Debug tab is the place to inspect that API the same way an external client would.

## What the Debug tab gives you

- A serve panel with the public API status, URL, and quick start/stop controls
- An API editor that can send real requests against `http://127.0.0.1:8800/v1`
- Built-in examples for common flows:
  - `GET /v1/health`
  - `GET /v1/models`
  - `GET /v1/models/{model}`
  - `POST /v1/models/stats`
  - `GET /v1/context/status`
  - `GET /v1/sessions`
  - `POST /v1/models/load`
  - `POST /v1/models/unload`
  - `POST /v1/chat/completions`
- A recent-request list for fast replays
- A live logs console
- Raw prompt and parse trace views

## Recommended flow

When you want to test the local API end to end, this is the simplest sequence:

1. Run `GET /v1/models` to confirm the registry sees your models.
2. Run `GET /v1/models/{model}` if you want the full metadata for one model.
3. Run `POST /v1/models/load` with the model filename you want.
4. Poll `POST /v1/models/stats` with that same model name until the load reaches `ready`.
5. Run `POST /v1/chat/completions` with a normal OpenAI-compatible body.

The Models tab and the API editor should reflect the same underlying state.

## Public API vs Direct App

The Debug workspace prefers the public HTTP API first so you can test the exact surface external tools use.

If the public API is unavailable, the editor can fall back to a direct in-process route for supported endpoints. That keeps the Debug tab usable while still making it obvious whether you hit:

- `Public API`
- `Direct App`

If you are validating external integrations, always prefer `Public API`.

## cURL examples

List models:

```bash
curl "http://127.0.0.1:8800/v1/models"
```

Inspect one model:

```bash
curl "http://127.0.0.1:8800/v1/models/Your-Model-Here.gguf"
```

Load a model:

```bash
curl -X POST "http://127.0.0.1:8800/v1/models/load" \
  -H "Content-Type: application/json" \
  --data-raw "{\"model\":\"Your-Model-Here.gguf\"}"
```

Check load progress:

```bash
curl -X POST "http://127.0.0.1:8800/v1/models/stats" \
  -H "Content-Type: application/json" \
  --data-raw "{\"model\":\"Your-Model-Here.gguf\"}"
```

Send a chat completion:

```bash
curl -X POST "http://127.0.0.1:8800/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-raw "{\"model\":\"Your-Model-Here.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: InferenceBridge ready\"}],\"stream\":false}"
```

## Notes on loading

`POST /v1/models/load` is asynchronous.

That means a successful `200` response usually means:

- the request was accepted
- the backend started loading

It does not mean the model is already ready for generation. Use `POST /v1/models/stats` with the same model name, the Models tab, or the footer status to confirm readiness.

## Notes on model names

The API uses model filenames as IDs right now. The easiest reliable pattern is:

1. call `GET /v1/models`
2. copy the `id`
3. optionally inspect `GET /v1/models/{id}`
4. use that same value in `POST /v1/models/load`, `POST /v1/models/stats`, or `POST /v1/chat/completions`

## Troubleshooting

If the serve badge is not running:

- Start the API from the Debug header or Settings
- Check the Logs tab for bind errors
- If port `8800` is occupied by a stale `llama-server` or another InferenceBridge instance, use the app banner action to kill the stale owner when offered

If a chat request fails:

- Make sure a model is loaded, or include a valid `model` field
- Check `/v1/models` first
- Check `POST /v1/models/stats` with the same model name to make sure the model finished loading

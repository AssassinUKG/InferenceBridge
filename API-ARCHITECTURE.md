# InferenceBridge API Architecture

## Single-Port Design

InferenceBridge exposes **one port** to the outside world. The internal
llama-server process runs on an ephemeral port that is auto-assigned by the
OS at launch time — clients never need to know or care about it.

```
External clients (HelixClaw, curl, etc.)
        |
        v
  +--------------------------+
  |  InferenceBridge Axum    |  <-- Port 8800 (config: server.port)
  |  API Server              |      OpenAI-compatible + IB extensions
  |                          |      + transparent llama-server proxy
  +--------------------------+
        |  proxies to
        v
  +--------------------------+
  |  llama-server            |  <-- Ephemeral port (auto-assigned, internal only)
  |  (managed child process) |      Native llama.cpp endpoints
  +--------------------------+
```

- **Port 8800** (`server.port`): The single public-facing Axum HTTP server.
  Handles authentication, OpenAI-compatible chat/completion routing, model
  load/unload lifecycle, InferenceBridge extensions, **and** transparently
  proxies all llama-server native endpoints (`/props`, `/slots`, `/tokenize`,
  `/detokenize`, `/embedding`, etc.).

- **Ephemeral internal port**: The llama-server child process binds to a
  free port chosen by the OS at launch time (port 0 → OS picks). This port
  is never exposed in config, never user-configurable, and never visible to
  external clients. Every handler reads `process.port()` at request time.

## API Routes

### OpenAI-Compatible (`/v1/...`)

| Method | Path                      | Handler                     | Description                          |
|--------|---------------------------|-----------------------------|--------------------------------------|
| POST   | `/v1/chat/completions`    | `completions::chat_completions` | Chat completion (streaming + non-streaming) |
| POST   | `/v1/completions`         | `completions::text_completions` | Text completion                      |
| POST   | `/v1/responses`           | `responses::responses`      | OpenAI Responses API                 |
| GET    | `/v1/models`              | `models::list_models`       | List all available models            |
| GET    | `/v1/models/:model`       | `models::get_model`         | Get model details (context, quant, capabilities) |
| POST   | `/v1/models/load`         | `models::load_model`        | Load a model (with optional `context_size`) |
| POST   | `/v1/models/unload`       | `models::unload_model`      | Unload the current model             |
| GET    | `/v1/models/stats`        | `models::current_model_stats` | Current model runtime stats        |
| POST   | `/v1/models/stats`        | `models::model_stats`       | Stats for a specific model           |
| GET    | `/v1/health`              | `health::health_check`      | Health check (model status + KV cache) |
| GET    | `/v1/metrics`             | `metrics::get_metrics`      | Inference metrics (tokens/sec, etc.) |
| POST   | `/v1/inference/cancel`    | `metrics::cancel_inference`  | Cancel active inference              |

### InferenceBridge Extensions (`/v1/...`)

| Method | Path                      | Handler                     | Description                          |
|--------|---------------------------|-----------------------------|--------------------------------------|
| GET    | `/v1/context/status`      | `extensions::context_status` | KV cache fill ratio + token counts  |
| GET    | `/v1/runtime/status`      | `extensions::runtime_status` | Full runtime status                 |
| GET    | `/v1/debug/profile`       | `extensions::debug_profile`  | Debug/profiling info                |
| GET    | `/v1/sessions`            | `extensions::list_sessions`  | List chat sessions                  |
| POST   | `/v1/sessions`            | `extensions::create_session` | Create a new session                |
| DELETE | `/v1/sessions/:id`        | `extensions::delete_session` | Delete a session                    |
| GET    | `/v1/sessions/:id/messages` | `extensions::get_session_messages` | Get session message history |

### Native API (`/api/v1/...`)

Subset of routes for programmatic access (same handlers):

| Method | Path                  | Handler                |
|--------|-----------------------|------------------------|
| GET    | `/api/v1/models`      | `models::list_models`  |
| POST   | `/api/v1/models/load` | `models::load_model`   |
| POST   | `/api/v1/models/unload` | `models::unload_model` |
| GET    | `/api/v1/health`      | `health::health_check` |

### Transparent llama-server Proxy (root level)

Any request that does **not** match `/v1/*` or `/api/v1/*` is transparently
forwarded to the internal llama-server process. This means every native
llama-server endpoint works through InferenceBridge automatically:

| Path          | Description                                            |
|---------------|--------------------------------------------------------|
| `/props`      | Server properties (`n_ctx`, generation settings)       |
| `/slots`      | Slot info (KV cache per-slot)                          |
| `/tokenize`   | Tokenize text                                          |
| `/detokenize` | Detokenize tokens                                      |
| `/embedding`  | Generate embeddings                                    |
| `/health`     | Native llama-server health (distinct from `/v1/health`)|
| `/*`          | Any future llama-server endpoint                       |

These endpoints bypass API key authentication (matching native llama-server
behavior).

## Request Flow

### Chat Completion

```
Client POST /v1/chat/completions
  -> Axum handler (completions.rs)
  -> Reads internal port from AppState.process.port()
  -> Creates LlamaClient(internal_port)
  -> Translates OpenAI format to llama-server /completion format
  -> Forwards to http://127.0.0.1:{internal_port}/completion
  -> Translates response back to OpenAI format
  -> Returns to client
```

### Model Load

```
Client POST /v1/models/load {"model": "...", "context_size": 32768}
  -> models::load_model handler
  -> Calls backend_load_model() in commands/model.rs
  -> Shuts down current llama-server (if running)
  -> Launches new llama-server with --ctx-size 32768
     (OS auto-assigns ephemeral port)
  -> Reads actual port from process.port()
  -> Waits for /health on internal port to return 200
  -> Queries /slots or /props for actual context length
  -> Updates AppState (model_stats, loaded_model)
  -> Returns load result to client
```

### Context Detection (via /props transparent proxy)

```
Client GET /props
  -> Axum fallback handler (transparent proxy)
  -> Reads internal port from AppState.process.port()
  -> Forwards GET to http://127.0.0.1:{internal_port}/props
  -> Returns raw JSON (includes default_generation_settings.n_ctx)
```

## Authentication

Bearer token authentication is enabled when `config.server.api_key` is set.
All routes require the `Authorization: Bearer <key>` header except:

- `GET /v1/health` and `GET /api/v1/health`
- Any root-level path (transparently proxied to llama-server)
- `OPTIONS` (CORS preflight)

## Configuration

Relevant config fields in `inference-bridge.toml`:

```toml
[server]
port = 8800           # Public API port (the only external port)
host = "127.0.0.1"    # Bind address
autostart = true       # Start API server on app launch
api_key = ""           # Bearer token (empty = no auth)
```

There is no `backend_port` configuration. The internal llama-server port is
auto-assigned by the OS at launch time and managed entirely by InferenceBridge.

## Integration with HelixClaw

HelixClaw can connect to InferenceBridge using the `llamacpp` provider:

```toml
# helixclaw config
provider = "llamacpp"
base_url = "http://127.0.0.1:8800"
```

The warm-up flow:
1. HelixClaw sends `POST /v1/models/load` with `context_size` (best-effort, OK if not IB)
2. IB launches llama-server with `--ctx-size` on an auto-assigned internal port
3. HelixClaw queries `GET /props` (transparently proxied to llama-server)
4. Actor context_limit is set to the real value from llama-server

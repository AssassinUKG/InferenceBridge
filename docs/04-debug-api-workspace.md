# InferenceBridge API Reference

Base URL: `http://127.0.0.1:8800` (configurable via `server.port` in `inference-bridge.toml`)

All endpoints are available under both `/v1` (OpenAI-compatible) and `/api/v1` (native) prefixes where noted.

## Authentication

When `server.api_key` is set in config, all requests (except health checks) require:

```
Authorization: Bearer <your-api-key>
```

Health checks and CORS preflight (`OPTIONS`) always pass through.

---

## Health & Monitoring

### GET /v1/health

Available on both `/v1/health` and `/api/v1/health`.

Returns server health status with KV cache metrics.

```bash
curl http://127.0.0.1:8800/v1/health
```

**Response:**

```json
{
  "status": "ok",
  "model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "kv_cache": {
    "total_tokens": 32768,
    "used_tokens": 1024,
    "fill_ratio": 0.03
  }
}
```

Status values: `"ok"`, `"unhealthy"`, `"no_model"`.

### GET /v1/metrics

Cumulative metrics and current state.

```bash
curl http://127.0.0.1:8800/v1/metrics
```

**Response:**

```json
{
  "model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "model_load_state": "loaded",
  "context_size": 32768,
  "last_load_duration_ms": 4200,
  "last_inference": {
    "source": "api",
    "model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "started_at": "2026-03-28T08:48:00Z",
    "finished_at": "2026-03-28T08:48:05Z",
    "elapsed_ms": 5000,
    "time_to_first_token_ms": 320,
    "prompt_tokens": 150,
    "completion_tokens": 200,
    "total_tokens": 350,
    "prompt_tokens_per_second": 468.75,
    "decode_tokens_per_second": 42.5,
    "end_to_end_tokens_per_second": 40.0
  },
  "cumulative": {
    "total_requests": 42,
    "total_errors": 1,
    "total_cancellations": 0,
    "total_model_loads": 3,
    "total_model_unloads": 2,
    "backend_restart_count": 0
  },
  "process_state": "Running",
  "uptime_secs": 1711612091
}
```

### GET /v1/runtime/status

Full runtime status including process info, scheduler state, and last launch config.

```bash
curl http://127.0.0.1:8800/v1/runtime/status
```

**Response:**

```json
{
  "state": "Running",
  "model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "previous_model": null,
  "model_load_state": "Loaded",
  "model_load_progress": null,
  "active_generation": null,
  "crash_count": 0,
  "server_version": "b5200",
  "server_path": "C:\\...\\llama-server.exe",
  "backend": "cuda",
  "api_state": "Running",
  "api_error": null,
  "api_url": "http://127.0.0.1:8800/v1",
  "api_reachable": true,
  "api_port_owner": null,
  "startup_duration_ms": 4200,
  "parallel_slots": 1,
  "slot_count": 1,
  "active_requests": 0,
  "queued_requests": 0,
  "scheduler_limit": 1,
  "last_launch_preview": { ... },
  "last_generation_metrics": { ... }
}
```

### GET /v1/context/status

Live KV cache context status from the running llama-server.

```bash
curl http://127.0.0.1:8800/v1/context/status
```

**Response:**

```json
{
  "total_tokens": 32768,
  "used_tokens": 512,
  "fill_ratio": 0.015,
  "pinned_tokens": 0,
  "rolling_tokens": 0,
  "compressed_tokens": 0,
  "last_compaction_action": null
}
```

---

## Models

### GET /v1/models

Available on both `/v1/models` and `/api/v1/models`.

Lists all scanned GGUF models (OpenAI-compatible format with extensions).

```bash
curl http://127.0.0.1:8800/v1/models
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
      "object": "model",
      "created": 1711612091,
      "owned_by": "inference-bridge",
      "active": true,
      "max_context_length": 262144,
      "state": "loaded",
      "reasoning": {
        "supported": true,
        "separates_content": true,
        "effort_values": ["none", "low", "medium", "high", "xhigh"],
        "supports_reasoning_tokens": true,
        "default_effort": "medium"
      }
    }
  ]
}
```

### GET /v1/models/:model

Full model detail by filename.

```bash
curl http://127.0.0.1:8800/v1/models/Qwen3.5-35B-A3B-Q4_K_M.gguf
```

**Response:**

```json
{
  "id": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "object": "model",
  "created": 1711612091,
  "owned_by": "inference-bridge",
  "active": true,
  "path": "C:\\Users\\...\\Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "size_bytes": 19456789012,
  "size_gb": 18.12,
  "family": "Qwen3.5",
  "supports_tools": true,
  "supports_reasoning": true,
  "supports_vision": false,
  "context_window": 32768,
  "max_context_length": 262144,
  "max_output_tokens": 8192,
  "quant": "Q4_K_M",
  "tool_call_format": "Qwen",
  "think_tag_style": "Standard",
  "reasoning": { ... }
}
```

---

## Model Loading

### POST /v1/models/load

Available on both `/v1/models/load` and `/api/v1/models/load`.

Load a model into the llama-server backend. The context size you send is passed directly to `--ctx-size` — no defaults are injected.

**Request:**

```bash
curl -X POST http://127.0.0.1:8800/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-35B-A3B-Q4_K_M.gguf", "context_size": 32768}'
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | yes | Model filename or path-style name (e.g. `"qwen/qwen3.5-4b"`) |
| `context_size` | u32 | no | Context window size passed to `--ctx-size`. Omit to use the model's GGUF metadata default. |
| `echo_load_config` | bool | no | If `true`, response includes `load_config` with the effective context. |

**Context size aliases** — all of these map to the same field:

`context_size`, `contextLength`, `context_length`, `contextlength`, `ctx_size`, `n_ctx`, `maxContextLength`, `num_ctx`, `numCtx`

Also extracted from nested objects: `load_config.context_size`, `options.num_ctx`, etc.

**Behaviour:**

- If the same model is already loaded with the same context, the request returns immediately (coalesce).
- If the same model is loaded but with a different context, it **reloads** with the new context.
- If no `context_size` is sent, llama-server uses the GGUF metadata default.
- Path-style model names (e.g. `"qwen/qwen3.5-4b"`) are resolved by extracting the last segment and matching against scanned filenames.

**Response:**

```json
{
  "type": "llm",
  "instance_id": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "load_time_seconds": 4.2,
  "status": "loaded",
  "load_config": {
    "context_length": 32768
  },
  "model_info": {
    "model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "context_size": 32768,
    "tokens_per_sec": 0.0,
    "memory_mb": 0
  }
}
```

`load_config` is only present when `echo_load_config: true`. `model_info` always reflects the current model stats.

### POST /v1/models/unload

Available on both `/v1/models/unload` and `/api/v1/models/unload`.

Unload the current model and stop the llama-server backend.

```bash
curl -X POST http://127.0.0.1:8800/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-35B-A3B-Q4_K_M.gguf"}'
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | no | Model to unload. If omitted, unloads whatever is currently loaded. |
| `instance_id` | string | no | Alias for `model`. |

**Response:**

```json
{
  "instance_id": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "status": "unloaded"
}
```

### GET /v1/models/stats

Returns stats for the currently loaded model.

```bash
curl http://127.0.0.1:8800/v1/models/stats
```

### POST /v1/models/stats

Returns stats for a specific model.

```bash
curl -X POST http://127.0.0.1:8800/v1/models/stats \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-35B-A3B-Q4_K_M.gguf"}'
```

**Response:**

```json
{
  "requested_model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "active_model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "matches_active_model": true,
  "state": "Loaded",
  "progress": null,
  "stats": {
    "model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "context_size": 32768,
    "tokens_per_sec": 42.5,
    "memory_mb": 8192
  },
  "model": { ... }
}
```

---

## Inference

### POST /v1/chat/completions

OpenAI-compatible chat completions. Supports streaming and non-streaming.

```bash
curl -X POST http://127.0.0.1:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | | Model filename. If omitted, uses the currently loaded model. |
| `messages` | array | | OpenAI-format message array. |
| `stream` | bool | `false` | Enable SSE streaming. |
| `max_tokens` | u32 | | Max tokens to generate. Aliases: `max_completion_tokens`, `maxTokens`. |
| `temperature` | f32 | | Sampling temperature. Alias: `temp`. |
| `top_p` | f32 | | Nucleus sampling. Alias: `topP`. |
| `top_k` | i32 | | Top-K sampling. Alias: `topK`. |
| `min_p` | f32 | | Min-P sampling. Alias: `minP`. |
| `presence_penalty` | f32 | | Alias: `presencePenalty`. |
| `frequency_penalty` | f32 | | Alias: `frequencyPenalty`. |
| `repetition_penalty` | f32 | | Aliases: `repetitionPenalty`, `repeatPenalty`, `repeat_penalty`. |
| `seed` | i64 | | Reproducibility seed. |
| `stop` | string/array | | Stop sequence(s). |
| `tools` | array | | OpenAI-format tool definitions. |
| `context_size` | u32 | | If sent, triggers model reload with this context. Same aliases as load endpoint. |
| `reasoning` | object | | `{ "effort": "medium", "max_tokens": 4096 }` |
| `reasoning_effort` | string | | Shorthand: `"none"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`. |
| `stream_options` | object | | `{ "include_usage": true }` — include token usage in final SSE chunk. |
| `options` | object | | Ollama-format options (e.g. `{ "num_ctx": 32768 }`). |

**Messages format:**

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello"},
  {"role": "user", "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}
]
```

**Non-streaming response:**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1711612091,
  "model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

**Streaming:** Returns SSE events (`data: {...}\n\n`) with `delta.content` chunks, ending with `data: [DONE]`.

### POST /v1/completions

Legacy text completions endpoint. Same parameters as chat completions but uses `prompt` instead of `messages`.

### POST /v1/responses

OpenAI Responses API format. Accepts `input` as either a plain string or a messages array.

```bash
curl -X POST http://127.0.0.1:8800/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "input": "Explain quantum computing briefly",
    "stream": false
  }'
```

| Field | Type | Description |
|-------|------|-------------|
| `input` | string or array | Plain text or OpenAI message array. |
| `max_output_tokens` | u32 | Alias: `maxOutputTokens`. |

### POST /v1/inference/cancel

Cancel the currently active inference request.

```bash
curl -X POST http://127.0.0.1:8800/v1/inference/cancel
```

**Response:**

```json
{
  "cancelled": true,
  "message": "Active inference request cancelled"
}
```

If nothing is running:

```json
{
  "cancelled": false,
  "message": "No active inference request to cancel"
}
```

---

## Sessions

### GET /v1/sessions

List all chat sessions.

```bash
curl http://127.0.0.1:8800/v1/sessions
```

### POST /v1/sessions

Create a new session.

```bash
curl -X POST http://127.0.0.1:8800/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"name": "My Session", "model_id": "Qwen3.5-35B-A3B-Q4_K_M.gguf"}'
```

### DELETE /v1/sessions/:id

Delete a session by ID.

```bash
curl -X DELETE http://127.0.0.1:8800/v1/sessions/abc123
```

**Response:**

```json
{ "deleted": true, "id": "abc123" }
```

### GET /v1/sessions/:id/messages

Retrieve all messages for a session.

```bash
curl http://127.0.0.1:8800/v1/sessions/abc123/messages
```

---

## Debug

### GET /v1/debug/profile

Inspect the effective model profile (after overrides).

```bash
curl "http://127.0.0.1:8800/v1/debug/profile?model=Qwen3.5-35B-A3B-Q4_K_M.gguf"
```

Returns the resolved profile including family, tool format, think tag style, context windows, and any user overrides applied.

---

## Configuration Reference

All values come from `inference-bridge.toml`. No hardcoded defaults are injected into API requests.

### `[server]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `port` | u16 | `8800` | Public API port. |
| `host` | string | `"127.0.0.1"` | Bind address. |
| `autostart` | bool | `true` | Start API server on app launch. |
| `default_temperature` | f32 | none | Server-level default (overridden per-request). |
| `default_top_p` | f32 | none | Server-level default. |
| `default_top_k` | i32 | none | Server-level default. |
| `default_max_tokens` | u32 | none | Server-level default. |
| `default_ctx_size` | u32 | none | Server-level default context for loads. |
| `api_key` | string | none | Bearer token required on all requests. |

### `[models]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `scan_dirs` | array | `[]` | Directories to scan for `.gguf` files. |

### `[process]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llama_server_path` | string | `""` | Path to llama-server. Empty = auto-detect. |
| `gpu_layers` | i32 | `-1` | GPU layers. `-1` = all on GPU. |
| `threads` | u32 | `0` | Generation threads. `0` = auto. |
| `threads_batch` | u32 | `0` | Batch threads. `0` = same as `threads`. |
| `kill_on_exit` | bool | `true` | Kill llama-server on app exit. |
| `backend_preference` | string | `"auto"` | `"auto"`, `"cuda"`, `"cpu"`. |
| `batch_size` | u32 | `0` | Prompt batch size (`-b`). `0` = default 2048. |
| `ubatch_size` | u32 | `0` | Micro-batch size (`-ub`). `0` = default 512. |
| `flash_attn` | bool | `false` | Enable Flash Attention (`-fa`). |
| `use_mmap` | bool | `true` | Memory-mapped model files. |
| `use_mlock` | bool | `false` | Lock model in RAM. |
| `cont_batching` | bool | `true` | Continuous batching (`-cb`). |
| `parallel_slots` | u32 | `1` | Parallel inference slots. |
| `main_gpu` | i32 | `0` | Primary GPU index for multi-GPU. |
| `defrag_thold` | f32 | `0.0` | KV cache defrag threshold. `0` = disabled. |
| `rope_freq_scale` | f32 | `0.0` | RoPE frequency scale. `0` = auto. |
| `backend_port` | u16 | `8801` | Internal llama-server port. |
| `model_load_timeout_secs` | u64 | `300` | Max time to wait for model load. |
| `first_token_timeout_secs` | u64 | `300` | Max time to first token during inference. |
| `inter_token_timeout_secs` | u64 | `120` | Max gap between tokens during inference. |
| `health_poll_interval_ms` | u64 | `150` | Health check polling interval during load. |

---

## Quick Start Flow

```bash
# 1. Check health
curl http://127.0.0.1:8800/v1/health

# 2. List available models
curl http://127.0.0.1:8800/v1/models

# 3. Load a model with specific context
curl -X POST http://127.0.0.1:8800/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-35B-A3B-Q4_K_M.gguf", "context_size": 32768}'

# 4. Check load progress
curl http://127.0.0.1:8800/v1/models/stats

# 5. Send a chat request
curl -X POST http://127.0.0.1:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# 6. Check performance metrics
curl http://127.0.0.1:8800/v1/metrics

# 7. Cancel if needed
curl -X POST http://127.0.0.1:8800/v1/inference/cancel
```

## Route Summary

| Method | Path | Also on `/api/v1` | Description |
|--------|------|--------------------|-------------|
| GET | `/v1/health` | yes | Health check + KV cache |
| GET | `/v1/models` | yes | List all models |
| GET | `/v1/models/:model` | | Model detail |
| POST | `/v1/models/load` | yes | Load model |
| POST | `/v1/models/unload` | yes | Unload model |
| GET | `/v1/models/stats` | | Current model stats |
| POST | `/v1/models/stats` | | Specific model stats |
| POST | `/v1/chat/completions` | | Chat completions |
| POST | `/v1/completions` | | Text completions |
| POST | `/v1/responses` | | Responses API |
| POST | `/v1/inference/cancel` | | Cancel active inference |
| GET | `/v1/metrics` | | Cumulative metrics |
| GET | `/v1/context/status` | | Live KV cache status |
| GET | `/v1/runtime/status` | | Full runtime status |
| GET | `/v1/debug/profile` | | Model profile debug |
| GET | `/v1/sessions` | | List sessions |
| POST | `/v1/sessions` | | Create session |
| DELETE | `/v1/sessions/:id` | | Delete session |
| GET | `/v1/sessions/:id/messages` | | Session messages |

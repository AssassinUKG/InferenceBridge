# InferenceBridge Implementation Plan

## Phase 1: Foundation

- Create the Tauri app shell
- Wire the React frontend to typed Tauri commands
- Establish shared app state and configuration loading
- Stand up the managed backend process layer

## Phase 2: Local Inference Core

- Model scanning and registry
- Model family and capability detection
- Managed `llama-server` lifecycle
- Shared load and unload paths for GUI and API
- Health checks and recovery behavior

## Phase 3: Sessions and Context

- SQLite session storage
- Message persistence
- Context status reporting
- Context rebuild and compression utilities

## Phase 4: Response Pipeline

- Prompt rendering
- Streaming support
- Think-tag cleanup
- Tool-call extraction
- JSON repair and output normalization

## Phase 5: Desktop UX

- Chat interface
- Model browser and loader
- Settings and process management
- Context panel
- Status bar
- Debug workspace

## Phase 6: Public API

- `GET /v1/health`
- `GET /v1/models`
- `GET /v1/models/{name}`
- `POST /v1/models/load`
- `POST /v1/models/unload`
- `POST /v1/models/stats`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- session and context endpoints

## Phase 7: Release Hardening

- Installer and release builds
- stale-process diagnostics
- single-instance behavior
- API serve controls
- documentation and examples
- CI validation

## Phase 8: Inference Runtime Roadmap

- Streaming and async execution hardening
- Model-aware configuration profiles and family-specific fixes
- Better context accounting, compaction, and memory handling
- Tighter `llama.cpp`-style execution control

See [docs/05-inference-runtime-roadmap.md](docs/05-inference-runtime-roadmap.md) for the detailed implementation plan.

## Success Criteria

- The app launches cleanly on supported platforms
- Models can be discovered, loaded, unloaded, and queried from both GUI and API
- The embedded API behaves like a stable OpenAI-compatible local endpoint
- The Debug tab can exercise the same API surface external clients use
- Sessions and context survive app restarts
- Public releases are documented, licensed, and validated in CI

# InferenceBridge Migration and Design Note

## Purpose

InferenceBridge is a desktop-first local inference app that combines:

- a Tauri desktop UI
- a shared OpenAI-compatible API
- managed `llama-server` lifecycle
- model-aware prompting and normalization
- session persistence and debugging tools

The goal is to make local inference feel like a single product instead of a loose collection of scripts, sidecars, and one-off integrations.

## Reuse Strategy

Several ideas from earlier internal experiments are still useful here:

| Area | Reuse strategy |
| --- | --- |
| Model profiles and family detection | Keep the capability-driven approach so model behavior can be tuned by family |
| Prompt rendering and template ownership | Keep prompt construction in-app instead of delegating it fully to the backend |
| Think-tag cleanup and JSON repair | Preserve the normalization pipeline for messy local model output |
| Retry and health checks | Keep explicit backoff and health probes around the backend process |
| Context tiering | Keep layered context management rather than a single unbounded prompt |

## What Not to Reuse

| Area | Reason |
| --- | --- |
| Multi-provider abstraction | InferenceBridge focuses on local GGUF inference rather than remote provider switching |
| LM Studio-specific logic | The app should manage its own process and API surface directly |
| Flat-file session storage | SQLite provides safer writes and better queries |
| Agent-specific orchestration logic | InferenceBridge is an inference layer and desktop app, not an agent runner |

## Core Design Decisions

### 1. One managed backend process

At any given time the app runs a single `llama-server` instance for the active model.

- model switch = unload current process, then start the next one
- simpler VRAM behavior
- fewer hidden background conflicts
- easier shared state between GUI and API

### 2. Shared state between GUI and API

The desktop UI and the public API are two interfaces over the same application state.

- loading from the GUI should be visible through the API
- loading from the API should be visible in the GUI
- unload, sessions, and context status should stay in sync

### 3. App-owned prompt and normalization pipeline

InferenceBridge owns the prompt path around the backend:

1. render model-aware prompt
2. send request to `llama-server`
3. normalize output
4. expose the result to GUI and API callers

This keeps behavior consistent across model families and makes debugging possible from inside the app.

### 4. SQLite-backed session persistence

SQLite is used for:

- sessions
- messages
- tool calls
- context snapshots

That gives better durability and better introspection than JSONL-only storage.

### 5. Local API as a first-class feature

The embedded API is not just a test hook. It is a supported part of the product.

- external tools should be able to rely on `http://127.0.0.1:8800/v1`
- the Debug tab should exercise that same surface
- serve lifecycle should be visible and controllable in the app

## Risk Areas

| Risk | Mitigation |
| --- | --- |
| Backend crash or stale process | explicit health checks, retries, and visible process controls |
| Port conflicts | surface owner diagnostics and offer safe kill actions for known stale owners |
| Model-specific formatting drift | keep template and normalization logic pluggable |
| Context overflow | track KV usage and trigger compression or rebuild when needed |
| Vision model misconfiguration | detect nearby projector files and warn when a text-only model is used with images |

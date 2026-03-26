# InferenceBridge Architecture

## High-Level Shape

InferenceBridge has three major layers:

1. React desktop UI
2. Rust application core
3. Managed `llama-server` backend

The UI and the public API both operate on the same Rust `AppState`.

## Top-Level Structure

```text
InferenceBridge/
|- src-tauri/
|  |- src/
|  |  |- api/
|  |  |- commands/
|  |  |- context/
|  |  |- engine/
|  |  |- models/
|  |  |- normalize/
|  |  |- session/
|  |  |- state.rs
|  |  |- config.rs
|  |  |- lib.rs
|  |  `- main.rs
|  `- tauri.conf.json
|- src/
|  |- components/
|  |- hooks/
|  |- lib/
|  `- App.tsx
`- docs/
```

## Request Flow

```text
User or external client
        |
        v
Desktop UI or /v1 API
        |
        v
Shared Rust AppState
        |
        v
Prompt rendering / model control / session store
        |
        v
Managed llama-server process
        |
        v
Normalization + persistence + response
```

## Main Subsystems

### UI

The React frontend handles:

- chat
- model discovery and loading
- settings
- context and serve status
- debug tools, logs, and API editor

### Commands

Tauri commands expose direct desktop actions for:

- chat
- sessions
- model load and unload
- settings
- context inspection
- debug data

### API

The Axum server provides the public `/v1` surface:

- health
- models
- load and unload
- stats
- chat and completions
- sessions
- context status

### Engine

The engine layer owns:

- `llama-server` discovery
- process launch and shutdown
- health polling
- request forwarding
- stream handling
- model-specific launch arguments

### Models

The models layer owns:

- directory scanning
- model registry
- family and capability detection
- active model tracking

### Normalize

The normalization pipeline handles:

- think-tag cleanup
- tool-call extraction
- JSON cleanup
- response shaping for API and UI consumers

### Session

SQLite stores:

- sessions
- messages
- tool calls
- related metadata used for replay or debugging

## Shared-State Principle

One of the key design goals is consistency:

- if the API loads a model, the GUI should reflect it
- if the GUI unloads a model, the API should reflect it
- serve status, context status, and session state should be shared rather than duplicated

That shared-state model is what makes the app behave like a single product instead of a GUI plus a separate sidecar.

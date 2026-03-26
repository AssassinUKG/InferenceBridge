# InferenceBridge

**A local LLM desktop app** — run any GGUF model on your own hardware with an OpenAI-compatible API and a clean, modern UI.

Built with Tauri (Rust) + React. Wraps `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp) and manages the whole thing — process lifecycle, model loading, streaming chat, API serving, and context monitoring.

---

## Screenshots

<!-- Chat interface with streaming response -->
![Chat](docs/screenshots/chat.png | width=100)

<!-- Model browser — search HuggingFace, download with progress -->
![Browse](docs/screenshots/browse.png | width=100)

<!-- Models tab — context slider, VRAM bar, per-family settings -->
![Models](docs/screenshots/models.png | width=100)

<!-- API inspector — live HTTP client for the local API -->
![API](docs/screenshots/api.png | width=100)

---

## Download

> **Pre-built installers available on the [Releases](../../releases) page.**

| Platform | Status |
|----------|--------|
| Windows (x64) | ✅ NSIS installer + MSI |
| macOS (Apple Silicon) | ✅ .dmg |
| Linux (Ubuntu/Debian x64) | ✅ .deb + AppImage |

**You do not need Rust, Node, or llama.cpp pre-installed.** InferenceBridge downloads and manages `llama-server` for you from within the app.

---

## Features

- **Browse & Download Models** — Search the full HuggingFace catalog for GGUF models, or pick from a curated list of top models. Download with live progress, see what's installed, delete local files.
- **Model Management** — Auto-detects model family (Qwen3.5, Qwen3, Llama3, DeepSeek R1, Phi, Mistral, Gemma…), applies per-family sampling defaults, context limits, and tool-call formats.
- **Chat** — Streamed completions, session history, per-session context, image support for vision models, thinking mode for Qwen3/DeepSeek R1.
- **OpenAI-Compatible API** — Drop InferenceBridge into any app that uses OpenAI's API. Works with Cursor, Continue, Open WebUI, SillyTavern, and more.
- **API Key Auth** — Optional Bearer token to secure the public endpoint.
- **Context Monitor** — Live KV-cache fill bar, auto-compression at 80%/95% thresholds.
- **API Inspector** — Interactive HTTP client for the local API — test completions, tool calls, list models, all from inside the app.
- **GPU & VRAM Monitoring** — Per-model VRAM estimates, live usage bar, spill-to-RAM visualisation.

---

## Quick Start (pre-built installer)

1. **Download** the installer for your OS from [Releases](../../releases) and run it.
2. **Open the app.** Go to **Settings → llama-server** and click **Download** (CUDA or CPU). The binary is fetched automatically.
3. **Add a model.** Go to **Browse** — search HuggingFace or pick from the curated list. Click **Download** on any quant.
4. **Load the model.** Switch to **Models**, click **Load**. A green dot appears in the header when it's ready.
5. **Chat** — or point any OpenAI-compatible client at `http://127.0.0.1:8800/v1`.

> Already have `.gguf` files on disk? Go to **Settings → Model Directories** and add your folder. InferenceBridge will pick them up immediately.

---

## Connecting Other Apps

InferenceBridge exposes a standard OpenAI-compatible endpoint once a model is loaded:

```
Base URL:  http://127.0.0.1:8800/v1
API Key:   (set one in Settings → API Key, or leave blank for open access)
```

**Cursor / VS Code Continue / any OpenAI client:**
```
Base URL: http://127.0.0.1:8800/v1
API Key:  your-key-here  (or anything if no key is set)
Model:    the filename of your loaded .gguf
```

**curl:**
```bash
curl http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key-here" \
  -d '{
    "model": "Qwen3-14B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/:id` | Get info for a specific model |
| `POST` | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| `POST` | `/v1/completions` | Text completion |
| `GET` | `/v1/context/status` | KV-cache fill / token usage |
| `GET` | `/v1/sessions` | List chat sessions |
| `POST` | `/v1/sessions` | Create a chat session |
| `DELETE` | `/v1/sessions/:id` | Delete a session |
| `GET` | `/v1/health` | Health check (no auth required) |

---

## Configuration

InferenceBridge stores its config at:

| OS | Path |
|----|------|
| Windows | `%APPDATA%\InferenceBridge\inference-bridge.toml` |
| macOS | `~/Library/Application Support/InferenceBridge/inference-bridge.toml` |
| Linux | `~/.config/InferenceBridge/inference-bridge.toml` |

All settings are also editable from the **Settings** tab in the GUI. See [`inference-bridge.example.toml`](./inference-bridge.example.toml) for a commented template.

### Key settings

```toml
[server]
port = 8800
host = "127.0.0.1"      # Change to "0.0.0.0" to expose on your network
api_key = ""             # Optional Bearer token — leave empty for open access
autostart = true         # Start the API server when the app launches

[models]
scan_dirs = [
    "C:\\Users\\You\\models",
]

[process]
gpu_layers = -1          # -1 = all layers on GPU, 0 = CPU only
threads = 0              # 0 = auto-detect
```

---

## Supported Model Families

| Family | Detection | Thinking Mode | Tool Calls |
|--------|-----------|--------------|------------|
| **Qwen3.5** | `qwen3.5` in filename | Qwen-style `/think` | QwenXml |
| **Qwen3** | `qwen3` in filename | Standard `<think>` | QwenXml |
| **Qwen2.5** | `qwen2.5` in filename | — | Native |
| **DeepSeek R1** | `deepseek` + `r1`/`reasoning` | Standard `<think>` | Native |
| **Llama 3.x** | `llama` + `3.`/`3-` | — | Hermes XML |
| **Phi 3/4** | `phi-3`/`phi-4` | — | Hermes XML |
| **Mistral / Mixtral** | `mistral`/`mixtral`/`nemo` | — | Hermes XML |
| **Gemma** | `gemma` in filename | — | Native |
| **Generic** | fallback | — | Native |

---

## Building from Source

You'll need:
- [Rust](https://rustup.rs/) 1.75+
- [Node.js](https://nodejs.org/) 18+
- Platform build tools:
  - **Windows:** Visual Studio Build Tools (C++)
  - **macOS:** Xcode Command Line Tools (`xcode-select --install`)
  - **Linux:** `sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev`

```bash
git clone https://github.com/your-org/InferenceBridge
cd InferenceBridge
npm install
npm run tauri build
```

Installer output: `src-tauri/target/release/bundle/`

### Development mode (hot reload)

```bash
npm run tauri dev
```

---

## Platform Notes

### macOS
- Apple Silicon (M1/M2/M3): full support, llama-server auto-downloaded
- Intel Mac: llama-server must be installed manually (`brew install llama.cpp`) or built from source

### Linux
- Ubuntu/Debian x64: llama-server auto-downloaded
- Other distros: install `llama-server` to your PATH or set the path in Settings
- Wayland: fully supported via Tauri's webview

### Windows
- CUDA (NVIDIA GPU): click "Download CUDA" in Settings → llama-server
- CPU only: click "Download CPU"
- AMD/Intel GPU: download the appropriate llama.cpp build manually and set the path in Settings

---

## Data Locations

| What | Location |
|------|----------|
| Config | See table above |
| Chat sessions (SQLite) | Same folder as config, `sessions.db` |
| Managed llama-server | `%LOCALAPPDATA%\InferenceBridge\bin\` (Windows) / `~/.local/share/InferenceBridge/bin/` |
| Downloaded models | First directory in `scan_dirs` |

---

## Architecture

```
┌─────────────────────────────────────────┐
│            React Frontend               │
│  Chat │ Models │ Browse │ Context │ API │
└──────────────────┬──────────────────────┘
                   │ Tauri IPC
┌──────────────────┴──────────────────────┐
│              Rust Backend               │
│  ┌───────────┐   ┌────────────────────┐ │
│  │ Commands  │   │   Axum API Server  │ │
│  │ (IPC cmds)│   │ /v1/* (port 8800)  │ │
│  └─────┬─────┘   └────────┬───────────┘ │
│        └─────────┬────────┘             │
│            ┌─────┴──────┐               │
│            │  AppState  │               │
│            └─────┬──────┘               │
│    ┌─────────────┼─────────────┐        │
│  Engine       Models        Session     │
│ (process,   (scanner,      (SQLite,     │
│  client,     registry,      export)     │
│  streaming)  profiles)                  │
│                 │                       │
│    ┌────────────┼────────────┐          │
│ Templates   Normalize    Context        │
│ (ChatML,    (think-strip, (tracker,     │
│  Qwen XML,   JSON repair,  strategy,    │
│  Llama3)     tool extract) compressor)  │
└──────────────────┬──────────────────────┘
                   │ HTTP :8801
            ┌──────┴──────┐
            │ llama-server │
            │  (llama.cpp) │
            └─────────────┘
```

---

## License

MIT

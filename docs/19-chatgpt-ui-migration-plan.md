# ChatGPT-Style UI Migration Plan

## Goal

Rebuild the InferenceBridge desktop interface around the quiet, content-first interaction model used by the ChatGPT desktop app while preserving InferenceBridge's local-model, runtime, API, model-management, and diagnostic capabilities.

This is a visual and interaction migration, not a product rebrand. InferenceBridge remains a local inference application and must continue to expose controls that ChatGPT does not need, including model loading, GGUF management, llama.cpp launch configuration, context state, and the OpenAI-compatible API.

## Scope Boundary

Voice is explicitly out of scope.

- No microphone button.
- No speech-to-text or recording state.
- No text-to-speech or read-aloud controls.
- No voice dependencies, backend services, or GPU allocation.

Chat supports text and image input only. Images must work through file selection, clipboard paste, and drag and drop, and must retain the existing vision-model readiness checks.

## Design Direction

- Persistent left app rail with primary destinations and chat history.
- Restrained dark surfaces, subtle borders, compact controls, and generous content spacing.
- One centered reading column for chat and focused settings content.
- Full-width operational workspaces for Models, Browse, Benchmark, Context, Logs, and API.
- Icon-first controls with tooltips for familiar actions.
- A single notification center rather than interruptive toast stacks.
- Shared buttons, icon buttons, fields, toggles, segmented controls, popovers, dialogs, and panels.
- Responsive behavior that remains usable in a narrow desktop window.

## Implementation Phases

### Phase 1 - Foundation and shell

Status: in progress

- Introduce shared visual tokens and reusable UI primitives.
- Replace the horizontal tab bar with a ChatGPT-style left navigation rail.
- Integrate chat history and the New chat action into that rail.
- Replace the old footer strip with a compact runtime status area.
- Keep notifications in the header bell panel.

### Phase 2 - Chat experience

Status: pending

- Center messages in a readable conversation column.
- Restyle user and assistant messages with quiet role treatment and message actions.
- Build a floating multiline composer with automatic height growth.
- Support image file selection, clipboard paste, and drag and drop.
- Show removable attachment previews and clear vision-readiness feedback.
- Move presets, thinking, and sampling into a compact controls popover.
- Preserve streaming text, collapsible reasoning, stop generation, keyboard send, and Escape-to-stop.
- Add clear empty, no-model, no-session, loading, and error states.

### Phase 3 - Operational workspaces

Status: pending

- Apply the shared shell and controls to Models and Browse.
- Put conservative 8K/16K load-context presets and predicted VRAM pressure directly beneath the selected model, while retaining the advertised model ceiling as an explicit advanced override.
- Keep the richer Hugging Face detail view, README preview, file sizes, download resume, and model deduplication.
- Restyle Benchmark, Context, Logs, and API as dense professional tools.
- Remove remaining one-off button, field, tab, and card treatments.

### Phase 4 - Settings

Status: pending

- Add a settings category rail and centered settings sections.
- Present settings as labelled rows with descriptions and controls aligned consistently.
- Keep advanced llama.cpp/runtime options discoverable without crowding common settings.

### Phase 5 - Verification and release

Status: pending

- Build the TypeScript/Vite frontend.
- Run Rust checks and compile tests.
- Exercise text chat, streaming, stop, image upload, image paste, image drop, and non-vision rejection.
- Check narrow and wide desktop layouts for clipping, overlap, and inaccessible controls.
- Build and launch the release executable for visual inspection.

## Completion Criteria

- Every primary page uses the same navigation, typography, spacing, controls, and status language.
- Chat is fully usable with text and images without exposing voice UI.
- Existing local inference and API behavior remains intact.
- Model management and Hugging Face workflows remain information-dense and functional.
- No major action depends on a transient toast or hidden hover-only control.
- The release build passes and the desktop app is visually checked at multiple window sizes.

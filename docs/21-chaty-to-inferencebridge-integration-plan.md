# Chaty UI Elements for InferenceBridge

**Status:** implemented as an initial IB test pass; UI work only  
**Scope:** the best Chaty chat-interface ideas that fit InferenceBridge  
**Source review:** [Chaty](https://github.com/Fangyuan025/Chaty) at commit [`23690f1`](https://github.com/Fangyuan025/Chaty/commit/23690f100a7d2653670901d7f218b7ec2033a735)  
**Related IB plan:** `docs/19-chatgpt-ui-migration-plan.md`

## Implementation snapshot (2026-07-17)

The selected UI slice is now implemented in IB without importing Chaty's agent loop or tool backend:

- safe rich Markdown with bounded highlighting, KaTeX, sanitized Mermaid, citation links and code preview actions;
- separate live/completed reasoning presentation, with hidden reasoning excluded from answer copy and export;
- conversation title search, persistent rename/pin, export and a keyboard command palette;
- clearer image attachment metadata, validation and vision-readiness/send-state feedback;
- truthful display-only tool activity cards backed by IB's existing normalized tool-call data;
- a sandboxed, CSP-restricted HTML/SVG Canvas with versions, copy, save, reload and bounded error reporting;
- responsive rail behaviour and operation-error feedback while preserving the existing IB workspace shell.

The implementation deliberately keeps the current composer inside `ChatPanel.tsx` and the conversation rail inside `AppSidebar.tsx`; those extractions were not required to deliver or safely test the behaviour. Feature flags were also not added because the features are presentation-only and have local failure fallbacks rather than alternate runtime paths.

Verification completed for this pass:

- frontend production build;
- 37 frontend/unit tests, including visible-only conversation export coverage;
- Rust library and test-target compilation;
- an isolated Windows Tauri release build, including shell-plugin capability validation;
- 1280 px browser inspection with no page-level horizontal overflow.

The 720 px and 480 px browser resize checks still need a native-app/manual pass because the controlled browser rejected viewport switching under its security policy. Rust test binaries compile, but execution remains subject to the repository's existing Windows native-loader issue.

## Decision

Adopt selected Chaty UI and interaction patterns, not Chaty's agent backend.

InferenceBridge keeps its existing:

- Rust runtime and shared application state;
- managed `llama-server` process and model/profile/settings logic;
- OpenAI and Anthropic APIs;
- tool-call normalization and repair;
- streaming, cancellation, context, replay and diagnostics;
- SQLite sessions;
- benchmark and readiness systems.

This work does **not** import or rebuild Chaty's frontend agent loop, file/shell/browser tools, approval authority, checkpoints, Deep Research loop, RAG engine, direct inference engine or model store.

Small supporting backend changes are allowed only where a visible UI feature needs persistence or an existing IB event exposed more cleanly—for example rename/pin/search/export commands. They must extend existing IB session/API modules and must not introduce a second tool or agent runtime.

## Best-only UI selection

| Priority | Chaty pattern | IB treatment | Chaty reference |
|---|---|---|---|
| P0 | Foldable reasoning with clear live/finished state | Adopt, driven by IB's existing reasoning stream rather than reparsing raw output | [`AssistantMessage.tsx`](https://github.com/Fangyuan025/Chaty/blob/main/src/components/AssistantMessage.tsx) |
| P0 | Rich Markdown, code copy, syntax highlighting, KaTeX, Mermaid and citations | Adopt with stricter sanitization and external-link handling | [`Markdown.tsx`](https://github.com/Fangyuan025/Chaty/blob/main/src/components/Markdown.tsx) |
| P0 | Clean conversation list with rename, pin, search and export | Adopt within IB's existing left rail and session store | [`App.tsx`](https://github.com/Fangyuan025/Chaty/blob/main/src/App.tsx) |
| P0 | Better composer attachments, paste/drop feedback and compact controls | Adapt to IB's currently supported text/image capabilities | [`App.tsx`](https://github.com/Fangyuan025/Chaty/blob/main/src/App.tsx) |
| P1 | Tool/action cards showing proposed action, arguments, result, failure and duration | Adopt as a renderer for IB's existing normalized/replay events; do not add execution | [`CodeMode.tsx`](https://github.com/Fangyuan025/Chaty/blob/main/src/components/CodeMode.tsx) |
| P1 | Searchable command palette for chats, models and common UI actions | Adopt with IB-native actions | [`CommandPalette.tsx`](https://github.com/Fangyuan025/Chaty/blob/main/src/components/CommandPalette.tsx) |
| P1 | Citation/source cards | Adopt when an IB response contains source metadata; hide otherwise | [`Markdown.tsx`](https://github.com/Fangyuan025/Chaty/blob/main/src/components/Markdown.tsx) |
| P2 | Sandboxed HTML Canvas preview with versions and runtime-error display | Adopt as an optional presentation feature for HTML code blocks | [`CanvasPanel.tsx`](https://github.com/Fangyuan025/Chaty/blob/main/src/components/CanvasPanel.tsx) |
| P2 | Compact progress/timeline presentation | Adapt for IB generation, reasoning, tool-call, model and error events | [`CodeMode.tsx`](https://github.com/Fangyuan025/Chaty/blob/main/src/components/CodeMode.tsx) |

## Explicit exclusions

Do not adopt these parts in this plan:

- Chaty's [`agentLoop.ts`](https://github.com/Fangyuan025/Chaty/blob/main/src/lib/agentLoop.ts);
- Chaty's Rust file, shell, browser, search, checkpoint or validation tools;
- frontend authority to approve or execute a mutation;
- the full separate **Code mode** product surface;
- Deep Research, knowledge-base, browser-driving or agent-plan UI without a real IB backend capability;
- voice/live mode;
- Chaty's direct llama.cpp/MLX model runtime or model downloader;
- Chaty's entire visual identity, layout or settings structure;
- loose Mermaid rendering or unsanitized generated SVG/HTML;
- monolithic components modelled after the very large `App.tsx` or `CodeMode.tsx` files.

If a backend capability is absent, its UI stays hidden. Do not ship decorative buttons, fake progress or controls that cannot complete an action.

## UI structure

Keep the current ChatGPT-style IB shell and introduce small, testable chat components:

```text
src/components/Chat/
  ChatPanel.tsx              # workspace composition only
  ChatHeader.tsx             # title and conversation actions
  ConversationRail.tsx       # list/search/pin/rename/export
  MessageList.tsx            # scrolling and virtualization boundary
  MessageBubble.tsx          # role layout and message actions
  MessageContent.tsx         # content-part selection
  MarkdownContent.tsx        # safe rich Markdown
  ReasoningPanel.tsx         # live and completed reasoning
  ToolActivityCard.tsx       # display only
  CitationCard.tsx
  Composer.tsx
  AttachmentTray.tsx
  CommandPalette.tsx
  CanvasPanel.tsx
```

Shared state should remain in focused hooks rather than move into one giant page component:

```text
src/hooks/
  useChat.ts
  useConversationList.ts
  useChatComposer.ts
  useChatShortcuts.ts
```

The UI consumes existing `MessageInfo`, stream/reasoning events, normalized tool calls and live-stream snapshots. Where those shapes are awkward, add a frontend view-model adapter. Do not fork IB's Rust parsing rules in React.

## Delivery plan

### Phase 0 - Preserve the current UI baseline

Before changing visible chat behaviour:

- capture wide, medium and narrow screenshots of the current Chat, Models, Browse, Benchmark, Context, Logs and API pages;
- record current text/image send, streaming, reasoning, stop, session switching and model switching behaviour;
- add lightweight component fixtures for user, assistant, system, reasoning, image, long code, table, malformed Markdown and tool-call messages;
- confirm the current UI migration in `docs/19-chatgpt-ui-migration-plan.md` remains the shell and styling authority;
- add UI-only flags for risky pieces: `rich_chat_rendering`, `tool_activity_cards`, `chat_canvas`.

Exit gate:

- `npm run build` and existing tests pass;
- screenshots and interaction notes make layout or behaviour regressions visible;
- no Rust runtime/API/tool-loop change is included.

### Phase 1 - Message rendering and reasoning

This is the highest-value first implementation slice.

Deliverables:

- extract the current reasoning UI into `ReasoningPanel.tsx`;
- show **Thinking…** while reasoning is live and **Reasoned** when complete;
- keep reasoning collapsed by default after completion, with user state retained for the active conversation;
- render reasoning from IB's separate reasoning stream/content field whenever available;
- retain orphan think-tag handling only as a legacy display fallback;
- add syntax highlighting with a bounded language bundle;
- add KaTeX for inline/block mathematics;
- add Mermaid diagrams using strict security mode or sanitized SVG;
- improve code-block language labels, copy feedback, wrapping and horizontal scrolling;
- add copy actions for visible answer text without copying hidden reasoning;
- support citation markers and source cards only when structured source data exists;
- ensure streaming Markdown remains stable without re-render flicker or scroll jumps.

Security rules:

- sanitize/validate `href` and image URLs;
- external links open through IB's validated external opener;
- reject `javascript:`, `file:` and unsafe data URLs;
- never insert raw model HTML into the main document;
- sanitize Mermaid SVG even when Mermaid reports valid output;
- place limits on diagram, math, code and table size before expensive rendering.

Likely files:

- `src/components/Chat/MessageBubble.tsx`;
- `src/components/Chat/MarkdownContent.tsx`;
- `src/components/Chat/StreamingText.tsx`;
- new `ReasoningPanel.tsx` and `CitationCard.tsx`;
- `src/styles/globals.css`;
- `package.json` only for reviewed rendering dependencies.

Exit gate:

- malicious Markdown/Mermaid/link fixtures cannot execute or navigate the Tauri window;
- long streamed answers remain responsive;
- reasoning never appears in copied final-answer text;
- plain Markdown and code remain readable when optional renderers fail.

Rollback: disable `rich_chat_rendering` and use the existing GFM renderer.

### Phase 2 - Conversation controls and command palette

Deliverables:

- add conversation search to the existing left rail;
- add inline rename with Enter/save and Escape/cancel;
- add pin/unpin and keep pinned chats above recent chats;
- add export for the active conversation using IB's existing export direction;
- retain delete confirmation and current New chat behaviour;
- add `Ctrl/Cmd+K` command palette covering:
  - New chat;
  - search/switch conversation;
  - load/switch/eject model through existing IB actions;
  - open Models, Browse, Benchmark, Context, Logs, API or Settings;
  - focus composer;
  - toggle reasoning visibility where supported;
- keep palette actions capability-aware and hide unavailable actions;
- add keyboard and screen-reader labelling for all conversation actions.

Minimal supporting data work allowed:

- add rename, pin and search methods to the existing session repository/commands;
- use SQLite FTS only if needed for message-content search; title filtering can remain frontend/local for the first slice;
- reuse the existing session export module rather than creating Chaty's database structure.

Exit gate:

- rename/pin/search survive restart;
- rapid session creation/switching does not select or rename the wrong chat;
- command palette never calls an unavailable or stale model/session action;
- no conversation data migration is destructive.

### Phase 3 - Composer and attachment polish

Deliverables:

- split the composer out of `ChatPanel.tsx`;
- preserve automatic textarea growth, Enter-to-send and Shift+Enter newline;
- preserve Escape-to-stop while streaming;
- improve file-picker, clipboard-paste and drag/drop states;
- show attachment preview, type, size, remove action and clear validation errors;
- show a direct vision-readiness explanation before sending an image to an incompatible runtime;
- keep sampling, thinking and generation presets in one compact controls popover;
- indicate active custom sampling without crowding the main composer;
- disable send with a precise reason during model load, incompatible vision state, active generation or empty input;
- keep unsupported file types out of the picker rather than accepting and failing them later.

Boundary:

- initial delivery remains text plus image because that is what IB currently supports;
- document/audio attachment controls are added only after a real IB ingestion path exists;
- no Chaty attachment parser is imported in this UI plan.

Exit gate:

- file select, paste, drop, remove, send and cancel work at narrow and wide sizes;
- an incompatible vision model never receives a hidden/stripped image;
- changing conversations clears only unsent state according to an explicit draft policy;
- current sampling/model-profile behaviour is unchanged.

### Phase 4 - Tool and generation activity presentation

This phase displays IB's current data. It does not create a tool executor or agent loop.

Deliverables:

- render normalized tool calls as compact activity cards with name, status and expandable arguments;
- render tool results/errors when IB already has them;
- show parser recovery/fallback as a subtle diagnostic indicator, not user-facing control markup;
- show duration and correlation/request ID in an advanced details section;
- visually separate model reasoning, visible answer, tool call and tool result;
- show generation errors inline at the correct turn;
- show a compact live activity indicator without exposing raw control tokens;
- allow copying sanitized arguments/results from expanded details;
- share presentation with Debug/API live-stream views where practical.

Do not add:

- approve/deny buttons unless a real backend approval request exists;
- execute/retry tool buttons that bypass current API semantics;
- artificial multi-step progress inferred from prose;
- frontend parsing of arbitrary text into executable actions.

Exit gate:

- native and recovered Qwen tool-call fixtures render equivalently;
- raw tool markers do not leak into the answer body;
- malformed/unknown calls remain truthful and do not look successfully executed;
- concurrent GUI/API stream events remain associated with the correct request.

Rollback: disable `tool_activity_cards`; retain current answer and debug displays.

### Phase 5 - Safe HTML Canvas preview

Deliverables:

- show **Preview** only for a fenced block confidently identified as standalone HTML/SVG;
- open a docked or modal Canvas panel without navigating away from IB;
- run content in a unique-origin sandboxed iframe;
- exclude native/Tauri bridge access and same-origin privileges;
- collect bounded runtime/resource errors via validated `postMessage` events;
- support in-memory version history for edits produced by later messages;
- support Copy, Save and Open externally through explicit user actions;
- show source beside preview and provide a reset/reload action.

Security rules:

- sanitize exported filenames and locations;
- validate message source and shape for iframe error events;
- do not grant `allow-same-origin`;
- do not expose local files, environment, tokens or application storage;
- restrict external navigation/popups and make them user-confirmed;
- impose HTML size and event-rate limits.

Exit gate:

- Canvas security fixtures cannot reach the Tauri bridge or main-window DOM;
- malformed pages fail inside the panel without affecting chat;
- closing the panel releases timers/resources;
- the feature is absent when `chat_canvas` is disabled.

### Phase 6 - Visual integration and release gate

Deliverables:

- align the adopted elements with IB's current typography, spacing, colours, controls and icons;
- keep Models, Browse, Benchmark, Context, Logs and API information density intact;
- keep the current left rail and operational workspaces rather than copying Chaty's whole shell;
- virtualize or progressively render long conversations if profiling shows a real need;
- add empty, loading, offline, no-model, incompatible-vision, interrupted-stream and retry states;
- complete keyboard navigation, focus restoration, reduced-motion and screen-reader review;
- check layouts at 1280, 720 and 480 px with no page-level horizontal overflow;
- run the Windows release build and visually inspect the actual Tauri app.

Exit gate:

- existing text/image chat, model loading, settings, APIs, benchmarks and diagnostics still work;
- no major action exists only on hover;
- optional Chaty-inspired components fail independently without breaking the chat screen;
- release build passes and the visible result looks like InferenceBridge, not a partial Chaty clone.

## UI test matrix

| Area | Required checks |
|---|---|
| Messages | User, assistant, system, image, reasoning, tool call/result, error, empty, long and malformed content. |
| Streaming | Text-only, reasoning+text, stop, failure, session switch, model switch and simultaneous API activity. |
| Markdown | GFM, code, tables, KaTeX, Mermaid, citations, huge blocks and renderer failure fallback. |
| Security | Script links, unsafe data URLs, SVG/HTML payloads, malicious Mermaid, iframe bridge access and popup/navigation attempts. |
| Conversations | Create, rename, pin, search, switch, export and delete across restart. |
| Composer | Keyboard send/newline, paste/drop/select/remove image, incompatible vision, disabled reasons and sampling controls. |
| Activity cards | Native/recovered/malformed calls, unknown tools, long arguments/results, errors and correct request association. |
| Accessibility | Keyboard-only use, focus visibility/restoration, labels, contrast, reduced motion and screen-reader reading order. |
| Responsive | 1280, 720 and 480 px; long code/table/citation; composer and rail open/closed. |
| Regression | Models, Browse, Benchmark, Context, Logs, API, Settings and runtime status remain usable. |

## Recommended implementation order

1. Reasoning panel and secure rich Markdown.
2. Conversation search/rename/pin/export and command palette.
3. Composer extraction and attachment polish.
4. Tool/generation activity cards using existing IB events.
5. Optional Canvas preview.
6. Accessibility, responsive and Windows release hardening.

The first code change should be Phase 1 only. It delivers the biggest visible improvement with the smallest risk to IB's current backend.

## Definition of done

This UI adoption is complete when:

- the best Chaty message, conversation, composer and navigation ideas feel native to IB;
- reasoning, tool calls, results, citations and errors are visually distinct and truthful;
- rich output is safe and has a plain fallback;
- conversation controls are persistent and keyboard accessible;
- only capabilities actually provided by IB are visible;
- no Chaty agent loop, tool executor or direct inference backend has been imported;
- IB's existing APIs, model runtime, profiles, benchmarks, sessions and diagnostics do not regress;
- every substantial UI addition can be disabled independently during rollout.

## Deferred

- Backend agent/tool-loop work;
- MCP execution and approvals;
- Mauler UI/integration plan;
- HelixClaw UI/integration plan;
- voice/live mode;
- Deep Research, RAG and browser automation.

Those can receive separate plans later if wanted. They are not prerequisites for this best-of-Chaty UI pass.

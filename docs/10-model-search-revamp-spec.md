# Model Search & Filter Revamp — Implementation Spec

**Goal:** Bring InferenceBridge's model discovery (Hugging Face hub search) and local model
library (My Models) up to LM Studio quality: server-side sort/filter, stable capability
facets, hardware-fit indicators per quant, lazy real file sizes, rich result metadata
(downloads / likes / updated), and a consistent badge system across the whole UI.

**Audience:** This document is a complete, self-contained brief for an implementing AI.
All file paths are relative to the repo root. Read the listed files before changing them.

---

## 1. Current state (what exists today)

| Area | File | Current behavior |
|---|---|---|
| HF search backend | `src-tauri/src/commands/browse.rs` | Single `search_hub_models(query, offset)` command. Hardcoded `sort=downloads`, `limit=20`, `filter=gguf`, `full=true`. No sort/author/capability params. |
| HF search UI | `src/components/Model/ModelBrowser.tsx` | Debounced (250 ms) search box, infinite scroll, tag chips **derived from the currently loaded page of results** (unstable, flicker as you scroll), tag filtering is **client-side over loaded results only**. No sort control. |
| Quant list | same | Quant rows show `size_gb` from HF siblings — but the HF list API does not return sibling sizes, so most show `0` and the size is hidden. No fit-on-GPU indicator. No recommended quant. |
| Downloads | same + `browse.rs` | Working download manager with progress, cancel, clear-done. Keep as-is. |
| Local library | `src/components/Model/ModelSelector.tsx` | "My Models" table with a top free-text search, a 4-item capability sidebar (all/reasoning/tools/vision/loaded), and **five free-text column filter inputs** (Arch/Params/LLM/Provider/Quant). No sort. No facet dropdowns. Substring-only matching. |
| Local tab in browser | `ModelBrowser.tsx` "Local" section | Flat list, no search/filter/sort at all. |
| GPU stats | `src/hooks/useGpuStats.ts`, `GpuStats` in `src/lib/types.ts` | `{ name, used_mb, dedicated_mb, free_mb, system_ram_mb }` polled live. Available for fit estimation. |
| Model type | `ModelInfo` in `src/lib/types.ts` | Rich: `family`, `quant`, `size_gb`, `supports_tools/reasoning/vision`, `context_window`, `max_context_window`, `hf_repo`, `n_layers`, `gguf_architecture`, etc. |

### Known defects to fix while you're in here
1. **Stale-response race** in `ModelBrowser.runSearch`: a slow earlier request can overwrite
   results of a later one. There is no request sequence guard or abort.
2. **Tag chips are unstable**: computed from `results`, so the chip row changes as pages load
   and a selected tag can vanish.
3. **Tag filter only filters loaded pages** — it should be a server-side filter (or at least
   trigger auto-fetch until enough matches exist).
4. `extract_quant` fallback returns garbage for filenames like `model-00001-of-00002.gguf`
   (returns `00002`). Multi-part GGUFs are also listed as separate quants instead of one
   logical entry.
5. Local table column filters are free-text substring inputs — replace with facet dropdowns.

---

## 2. Target UX (LM Studio parity checklist)

### 2.1 Hub search (Discover)
- [ ] Search box with debounce **and** race protection; Enter forces immediate search.
- [ ] **Sort dropdown**: Best match (HF relevance — omit `sort` param when query non-empty),
      Most downloaded, Most liked, Recently updated. Default: Most downloaded when query is
      empty, Best match when query is non-empty.
- [ ] **Capability filter chips (fixed set, always visible)**: All · Vision · Tools ·
      Reasoning · MoE · Coding. These map to server-side query strategies (§3.3), not
      client-side filtering of the loaded page.
- [ ] **Result cards** show: model name, publisher (with `owner/` prefix styled dimmer),
      downloads count (abbreviated: `1.2M`), likes count, "Updated 3w ago" relative date,
      params badge (`8B`), architecture badge (`llama`, `qwen3`…), capability badges,
      `Installed` badge when any quant is local, `Gated` badge for gated repos (shown but
      not downloadable, with a link-out hint).
- [ ] **Expanded card → quant table** with: quant label, real file size (lazy-fetched, §3.4),
      **fit indicator** (§3.5): green "Full GPU offload" / amber "Partial offload" /
      red "Likely too large", a ★ on the **recommended quant** (largest quant that fits
      fully in VRAM, preferring K-quants ≥ Q4), download button / progress / Installed state.
      Multi-part GGUFs (`-00001-of-0000N`) are collapsed into one row showing combined size,
      downloading all parts.
- [ ] "Open on Hugging Face" link per card.
- [ ] Skeleton loaders while searching (not just a centered text line).
- [ ] Keyboard: `/` or `Ctrl+F` focuses search; `↑/↓` moves card selection; `Enter`
      expands/collapses; `Esc` clears query.

### 2.2 Local library (My Models — `ModelSelector.tsx`)
- [ ] Keep layout (sidebar / table / inspector) but replace the five free-text column
      filters with **facet dropdowns** populated from actual model values:
      Arch (distinct `family || gguf_architecture`), Params (bucketed: <4B, 4–9B, 10–19B,
      20–39B, 40B+), Provider (distinct `modelPublisher()`), Quant (distinct `quant`).
      Each dropdown is a multi-select with counts, like `Q4_K_M (3)`.
- [ ] **Tokenized search**: split query on whitespace; every token must match at least one of
      filename / family / quant / hf_repo / provider (AND across tokens, OR across fields).
      Support `field:value` prefixes: `quant:q4`, `arch:qwen`, `params:8b`, `provider:bartowski`.
- [ ] **Sort control** on the table header: Name, Size, Params, Arch, Recently used (if
      available, else omit) — clickable column headers with ▲/▼ indicator.
- [ ] Sidebar capability filters gain **MoE** and counts stay live.
- [ ] **Fit indicator column/badge** per local model using the same estimator as §3.5
      (model `size_gb` vs free VRAM).
- [ ] Empty states must say *which* filters are active and offer one-click "Clear filters".

### 2.3 Local tab inside ModelBrowser
- [ ] Add the same tokenized search input + sort (Name/Size/Quant) to the "Local" section.
      Reuse the helpers from §4.3 — do not duplicate logic.

### 2.4 Unified badge system
- [ ] Create `src/components/common/Badge.tsx` exporting `<CapabilityBadge>`,
      `<QuantBadge>`, `<FitBadge>`, `<MetaBadge>` with one shared tone palette
      (move/merge `TAG_COLORS` from ModelBrowser and `MiniCap`/`CapBadge` from
      ModelSelector into it). Tones: reasoning=amber, tools=emerald, vision=pink,
      thinking=violet, moe=indigo, chat=cyan, math=orange, quant=amber-mono,
      fit-full=green, fit-partial=amber, fit-too-large=red, installed=cyan, gated=slate.
- [ ] Replace all inline badge implementations in `ModelBrowser.tsx`, `ModelSelector.tsx`
      (rows, inspector, LoadedModelRow) with these components. Visual style: rounded,
      10px uppercase tracking-wide, tinted bg + border at ~10–20% alpha as today.

---

## 3. Backend changes (`src-tauri/src/commands/browse.rs`)

### 3.1 New search command signature
Replace `search_hub_models(query, offset)` with a params struct (keep the old command as a
thin wrapper for one release if you want zero-risk, but update the UI to the new one):

```rust
#[derive(Debug, Deserialize)]
pub struct HubSearchParams {
    pub query: String,
    pub offset: u32,
    pub limit: u32,            // clamp 1..=50, default 20
    pub sort: Option<String>,  // "downloads" | "likes" | "lastModified" | None (=relevance)
    pub capability: Option<String>, // "vision" | "tools" | "reasoning" | "moe" | "coding"
    pub author: Option<String>,
}

#[tauri::command]
pub async fn search_hub_models_v2(params: HubSearchParams) -> Result<Vec<HubModel>, String>
```

### 3.2 Enriched `HubModel`
Extend the structs (additive — keep existing fields so the download flow is untouched):

```rust
pub struct HubModel {
    // existing: id, name, family, params, description, tags, supports_vision, quants
    pub downloads: u64,
    pub likes: u64,
    pub last_modified: Option<String>,   // ISO8601 from HF `lastModified`
    pub architecture: Option<String>,    // from gguf metadata or name heuristic
    pub params_b: Option<f32>,           // parsed parameter count in billions
    pub gated: bool,                     // surfaced instead of silently dropped
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub is_moe: bool,
}

pub struct HubQuant {
    // existing: quant, size_gb, url, filename
    pub parts: Vec<String>,   // all part filenames for multi-part GGUFs (len 1 normally)
}
```

Parsing notes:
- Add `last_modified: Option<String>` (`lastModified`) and `likes: u64` to `HfApiModel`.
  Both come back with `full=true`; also request `("config", "true")` is NOT needed.
- `params_b`: regex `(\d+(?:\.\d+)?)\s*[bB]\b` over `model_id`, falling back to the HF
  `safetensors.total` field if present (add `#[serde(default)] safetensors: Option<HfSafetensors>`
  with `total: Option<u64>`; params_b = total / 1e9).
- `architecture`: first match of a known-arch list (`llama, qwen, gemma, phi, mistral,
  mixtral, deepseek, glm, granite, smollm, command, starcoder, internlm, minicpm, olmo`)
  against lowercased `model_id`, else `None`.
- `supports_tools`: tag/name contains `tool`, `function-calling`, or arch in a known
  tools-capable set (qwen2.5+, qwen3, llama-3.x, mistral, command-r, glm-4, deepseek).
- `supports_reasoning`: name/tags contain `reason`, `thinking`, `r1`, `qwq`, `o1`, or
  arch in {deepseek-r1, qwq, qwen3, glm-z1}.
- `is_moe`: name/tags contain `moe`, `mixtral`, `-a\d+(\.\d+)?b` (e.g. `30B-A3B`), `x7b`/`x22b`.
- **Gated repos**: do NOT filter out in `hf_api_to_hub` anymore. Return them with
  `gated: true` and empty download affordance handled in UI. (Keep filtering
  `private || disabled`.)

### 3.3 Server-side capability filtering
HF's `/api/models` cannot filter "supports tools", so implement capability as a
**query strategy + post-filter**:
- `vision` → add HF query param `("pipeline_tag", "image-text-to-text")`, post-filter
  with existing `hf_supports_vision`.
- `tools` / `reasoning` / `moe` / `coding` → keep the user's text query, post-filter the
  page with the §3.2 detectors, and **over-fetch** (`limit * 3`, cap 60) so a page of 20
  survivors is likely. Return up to `limit` survivors. `hasMore` on the frontend stays
  driven by `found.length === limit`, so when over-fetching return exactly `limit` items
  if more survived, to keep pagination consistent. Track the *raw* HF offset on the
  frontend (return value isn't enough — see §4.1 `rawOffset` note).
- `coding` → append `coder OR code` heuristics: post-filter name contains `coder|code|starcoder|codestral|devstral`.

This is heuristic by design; precision over recall is fine. Document the limitation in a
code comment.

### 3.4 Real quant sizes (lazy per-repo fetch)
The list API returns siblings without sizes. Add:

```rust
#[tauri::command]
pub async fn get_hub_model_files(repo_id: String) -> Result<Vec<HubQuant>, String>
```

Implementation: `GET https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true`
(no auth needed for public repos), filter `.gguf`, map `size` (bytes) → `size_gb`, run the
same `extract_quant` + multi-part collapsing. The frontend calls this when a card is
expanded and merges sizes into the displayed quants (cache per repo in component state).
Timeout 15 s. On error, fall back to the sizes already present (often 0) and show "size
unknown" rather than failing the expansion.

**Multi-part collapsing** (apply in both `hf_api_to_hub` and `get_hub_model_files`):
group files matching `^(?P<stem>.+)-\d{5}-of-\d{5}\.gguf$` by `stem`; emit one `HubQuant`
with `filename = stem + ".gguf"` (display name), `parts` = all real filenames sorted,
`size_gb` = sum, `url` = url of part 1. The download handler in the frontend iterates
`parts`, calling `downloadHubModel` per part (sequentially is fine).

### 3.5 Fit estimation (frontend, no backend change)
Pure TS helper in `src/lib/fit.ts`:

```ts
export type FitLevel = "full" | "partial" | "too-large" | "unknown";
export function estimateFit(sizeGb: number, gpu: GpuStats | null): FitLevel
```

Rules (KV-cache headroom factor 1.2, matching the spirit of the existing VRAM estimator):
- `unknown` if `!gpu || sizeGb <= 0`.
- `full` if `sizeGb * 1.2 <= dedicated_mb / 1024`.
- `partial` if `sizeGb <= (dedicated_mb + system_ram_mb * 0.5) / 1024`.
- else `too-large`.

Used by: hub quant rows, recommended-quant star, local model rows. Tooltip text:
"Fits fully in VRAM (X.X GB free of Y GB)" / "Will partially offload to system RAM" /
"Exceeds VRAM + safe RAM overflow".

### 3.6 `extract_quant` fix
Before the suffix fallback, if the filename matches the multi-part pattern, strip the
`-NNNNN-of-NNNNN` segment first. If still no known quant token, return `"GGUF"` instead of
the last hyphen segment. Add unit tests:
`model-00001-of-00002.gguf → "GGUF"`, `Foo-Q4_K_M-00001-of-00002.gguf → "Q4_K_M"`.

### 3.7 Registration
Register the new commands (`search_hub_models_v2`, `get_hub_model_files`) in
`src-tauri/src/lib.rs` alongside the existing ones. Add TS wrappers in `src/lib/tauri.ts`
mirroring the structs (`HubSearchParams`, enriched `HubModel`, `HubQuant.parts`).

---

## 4. Frontend changes

### 4.1 `ModelBrowser.tsx` — search core
- Replace `runSearch` with a guarded version:
  ```ts
  const searchSeq = useRef(0);
  // inside runSearch: const seq = ++searchSeq.current;
  // after await: if (seq !== searchSeq.current) return;  // stale, drop
  ```
  Keep the 250 ms debounce; Enter submits immediately (cancel the pending timer).
- New state: `sort` ("relevance" | "downloads" | "likes" | "lastModified"),
  `capability` (null | "vision" | "tools" | "reasoning" | "moe" | "coding"),
  `rawOffset` (the HF API offset, which advances by the *requested page size including
  over-fetch*, returned alongside results — simplest: have `search_hub_models_v2` return
  `{ models, next_offset }` instead of a bare array. **Do this**; it removes all offset
  ambiguity from over-fetching).
- Changing sort or capability resets results and re-runs the search.
- Replace the dynamic tag chip row with the fixed capability chip row (§2.1). The
  per-result tag badges remain (rendered via the shared `CapabilityBadge`).
- Result card layout per §2.1. Relative-date helper: `timeAgo(iso: string)` in
  `src/lib/format.ts` (also move `formatBytes`, add `abbrevCount` for `1.2M`).
- On expand: call `getHubModelFiles(model.id)` once (cache in a `Map` ref), merge sizes,
  compute recommended quant = largest `size_gb` with `estimateFit(...) === "full"`,
  preferring quants matching `/^Q[45]_K/` on ties; mark with ★ and a "Recommended" label.
- Gated models: card shows `Gated` badge; quant rows show "Requires HF access" link to
  `https://huggingface.co/{id}` instead of a Download button.

### 4.2 Skeletons & keyboard
- Add a `ResultSkeleton` (3 shimmering rows) shown during initial load.
- Implement keyboard handling on the results container per §2.1 (roving `selectedIndex`
  state; `scrollIntoView({ block: "nearest" })` on change).

### 4.3 Shared local-filter helpers — `src/lib/modelFilter.ts`
```ts
export function tokenizeQuery(q: string): { field?: string; value: string }[]
export function matchesModel(m: ModelInfo, tokens: Token[]): boolean   // §2.2 semantics
export function modelParamsBucket(m: ModelInfo): string                // "<4B" | "4–9B" | ...
export function sortModels(ms: ModelInfo[], key: SortKey, dir: 1 | -1): ModelInfo[]
```
Move `modelParamsLabel`, `modelPublisher`, `shortModelName` out of `ModelSelector.tsx`
into this module and import them back (they're needed by both views).

### 4.4 `ModelSelector.tsx` — My Models table
- Replace `ColumnFilter` free-text inputs with a `FacetSelect` component (button showing
  `Arch ▾` / `Arch: qwen ✕`, opening a checklist popover with counts; multi-select;
  outside-click closes — copy the pattern from the existing downloads popover).
- Header cells become sort buttons (Name on the LLM column, Size needs a size column —
  add Size as a 7th column between Quant and Actions, showing `size_gb.toFixed(1)` GB
  and the `FitBadge`).
- Filtering pipeline: capability sidebar filter → facet selections → tokenized query →
  sort. All pure functions from §4.3, computed in a `useMemo`.
- Add `moe` to the sidebar `FILTERS` (detect via `is_moe`-style name heuristic on
  `ModelInfo.filename`/`family` — add `isMoeModel(m)` to `modelFilter.ts`).
- "No matches" empty state lists active filters and renders a "Clear all filters" button.

### 4.5 Badges
Implement `src/components/common/Badge.tsx` per §2.4 and sweep all usages:
- `ModelBrowser.tsx`: `TagBadge`, installed badge, quant labels, new fit/meta badges.
- `ModelSelector.tsx`: `MiniCap`, `CapBadge`, quant spans, Loaded/Installed pills.
Do not change the visual language (dark surface, tinted translucent fills) — consolidate it.

---

## 5. Out of scope (do NOT do)
- No HF authentication / token storage (gated downloads stay link-out only).
- No changes to the download engine, progress events, or delete flow.
- No virtualized lists (model counts are small).
- No persistence of search state across app restarts.
- No redesign of the inspector panel or load-options forms.

---

## 6. Acceptance criteria
1. Searching "qwen3", switching sort to "Recently updated", then quickly typing "llama"
   never shows qwen results after the llama response lands (race guard verified by
   throttling network in devtools).
2. Capability chip "Vision" returns only vision models across multiple scroll pages and
   the chip row never changes contents.
3. Expanding `bartowski/...` style repos shows non-zero sizes for every quant and exactly
   one row for multi-part GGUFs; downloading a multi-part entry fetches all parts.
4. Every quant row shows a fit badge consistent with `nvidia-smi` VRAM, and exactly one
   quant per card carries the ★ Recommended marker when any quant fully fits.
5. In My Models, `quant:q4 qwen` matches Qwen Q4_K_M models only; facet dropdowns show
   value counts; clicking a column header sorts and toggles direction.
6. Gated repos appear with a Gated badge and link-out, never a broken download.
7. `cargo test` passes including new `extract_quant` / multi-part / params-parse tests;
   `npm run build` (tsc) passes with no new errors.
8. No badge is rendered through an inline one-off implementation anymore — grep for
   `MiniCap|CapBadge|TagBadge` returns only `Badge.tsx` and imports.

## 7. Suggested implementation order
1. Backend: structs + `search_hub_models_v2` (returning `{ models, next_offset }`) +
   `get_hub_model_files` + `extract_quant`/multi-part fixes + tests. Register + TS wrappers.
2. `src/lib/fit.ts`, `src/lib/format.ts`, `src/lib/modelFilter.ts` (pure, unit-testable).
3. `Badge.tsx` + sweep.
4. ModelBrowser rewrite (search core → cards → quant table → skeletons → keyboard).
5. ModelSelector facets/sort/size column.
6. ModelBrowser Local tab search/sort.
7. Verify acceptance criteria end-to-end with `npm run tauri dev`.

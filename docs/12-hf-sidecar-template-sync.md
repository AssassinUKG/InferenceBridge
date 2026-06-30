# HF Sidecar And Template Sync

## Goal

Use Hugging Face only for small model sidecar files needed for correct local inference behavior, without downloading model weights during sync.

## Scope

- [x] Local GGUF launches stay local via `--model <path>`.
- [x] HF token is used only for metadata/sidecar/template requests.
- [x] Template sync populates the existing `hf-templates` cache.
- [x] Sidecar sync stores allowlisted config files under app support storage.
- [x] Model weight extensions are blocked by allowlist.
- [x] Sidecar fetches are capped at 2 MB per file.
- [x] UI button added: `Sync HF files`.
- [x] UI copy states that model weights are blocked.

## Allowed Files

- `chat_template.jinja`
- `tokenizer_config.json`
- `config.json`
- `generation_config.json`
- `special_tokens_map.json`

## Cache Locations

- Templates: `AppSupport/hf-templates/<repo>/...`
- Sidecars: `AppSupport/hf-sidecars/<repo>/...`

## Still To Do

- [x] Show per-model template cache status in model details.
- [x] Add a one-model sync action on individual model rows.
- [x] Add cache browser/open-folder action.
- [ ] Use cached tokenizer/config sidecars for richer model card/profile hints.

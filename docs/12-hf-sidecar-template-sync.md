# HF Model Metadata And Template Updates

## Goal

Use Hugging Face only for small model sidecar files needed for correct local inference behavior, without downloading model weights during sync.

## Scope

- [x] Local GGUF launches stay local via `--model <path>`.
- [x] HF token is used only for metadata/sidecar/template requests.
- [x] Template sync populates the existing `hf-templates` cache.
- [x] Sidecar sync stores allowlisted config files under app support storage.
- [x] Model weight extensions are blocked by allowlist.
- [x] Sidecar fetches are capped at 2 MB per file.
- [x] UI provides separate check and apply actions.
- [x] UI copy states that model weights are blocked.
- [x] Checks resolve an exact Hugging Face commit SHA instead of trusting a permanent `main` cache entry.
- [x] Quantized repos follow their declared `base_model:quantized:`/`base_model:` source when it publishes template metadata, while recording provenance.
- [x] Existing files are compared by content, so unrelated model-card or weight commits do not create false updates.
- [x] Templates embedded in `tokenizer_config.json` are extracted when no standalone Jinja file exists.
- [x] JSON and Jinja content is validated before activation.
- [x] Updates use immutable local snapshots, atomic file replacement, and one-step rollback.
- [x] A managed repo template overrides an outdated GGUF-embedded template on the next model load.

## Allowed Files

- `chat_template.jinja`
- `tokenizer_config.json`
- `config.json`
- `generation_config.json`
- `special_tokens_map.json`

## Cache Locations

- Active templates: `AppSupport/hf-templates/<repo>/chat_template.jinja`
- Active sidecars and update manifest: `AppSupport/hf-sidecars/<repo>/...`
- Immutable rollback snapshots: `AppSupport/hf-sidecars/<repo>/snapshots/<revision>/...`

## User Flow

1. Link the local GGUF to its Hugging Face source metadata if it is not already linked.
2. In Models, choose **Check HF files**. No active files change during this step.
3. If an update is reported, choose **Update HF files**.
4. Reload the model if it was already active; llama.cpp reads templates at process launch.
5. Use **Restore previous** if the new template is incompatible with that GGUF conversion.

The global **Check HF** action follows the same two-step behavior across all linked local models. A second click applies only sources that the previous check marked as changed.

## Safety Boundary

- The updater never follows repository filenames outside the allowlist.
- `.gguf`, `.safetensors`, tokenizer binaries, executable files, and arbitrary repository content cannot be fetched.
- Each accepted file is limited to 2 MiB before and after download.
- Downloads use the full remote commit SHA, preventing a branch from changing halfway through a snapshot.
- Invalid JSON, non-UTF-8 templates, empty templates, and content that does not resemble a messages-based Jinja template are rejected.
- If activation or manifest persistence fails, InferenceBridge restores the prior snapshot where one exists.

## Still To Do

- [x] Show per-model template cache status in model details.
- [x] Add a one-model sync action on individual model rows.
- [x] Add cache browser/open-folder action.
- [x] Detect actual upstream changes instead of treating any existing cache file as permanent.
- [x] Add rollback and active/remote revision status.
- [ ] Use cached tokenizer/config sidecars for richer model card/profile hints.

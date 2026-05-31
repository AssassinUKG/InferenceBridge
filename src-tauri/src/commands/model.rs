/// Emit a model-load-progress event to the GUI (no-op if handle is None / headless).
fn emit_load_progress_payload(
    handle: &Option<tauri::AppHandle>,
    progress: &crate::state::LoadProgress,
) {
    use tauri::Emitter;
    if let Some(h) = handle {
        let _ = h.emit("model-load-progress", progress.clone());
    }
}

fn emit_load_progress(
    handle: &Option<tauri::AppHandle>,
    stage: &str,
    msg: &str,
    progress: f32,
    done: bool,
    error: Option<String>,
) {
    emit_load_progress_payload(
        handle,
        &crate::state::LoadProgress {
            stage: stage.to_string(),
            message: msg.to_string(),
            progress,
            done,
            error,
        },
    );
}

fn done_model_load_state(
    transition: &crate::state::ModelLoadState,
    error: Option<&str>,
) -> crate::state::ModelLoadState {
    if let Some(error) = error {
        return crate::state::ModelLoadState::Error(error.to_string());
    }

    match transition {
        crate::state::ModelLoadState::Unloading => crate::state::ModelLoadState::Idle,
        _ => crate::state::ModelLoadState::Loaded,
    }
}

fn load_transition_for_request(
    current_model: Option<&str>,
    requested_model: &str,
) -> crate::state::ModelLoadState {
    match current_model {
        Some(current) if !names_match(current, requested_model) => {
            crate::state::ModelLoadState::Swapping
        }
        _ => crate::state::ModelLoadState::Loading,
    }
}

fn names_match(left: &str, right: &str) -> bool {
    let left = left.to_lowercase();
    let right = right.to_lowercase();
    left == right
        || left.trim_end_matches(".gguf") == right
        || left == right.trim_end_matches(".gguf")
}

fn normalize_requested_context_size(context_size: Option<u32>) -> Option<u32> {
    context_size.filter(|value| *value > 0)
}

/// Resolve the context size to pass to llama-server.
///
/// Returns only what was explicitly requested. No defaults, no fallbacks.
/// If nothing was requested, returns `None` and llama-server uses whatever
/// the GGUF model metadata specifies.
fn resolve_launch_context_size(requested_context_size: Option<u32>) -> Option<u32> {
    normalize_requested_context_size(requested_context_size)
}

fn sanitize_hf_cache_segment(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

async fn ensure_cached_hf_template(
    repo_id: &str,
    template_path: &str,
) -> Result<std::path::PathBuf, String> {
    let template_relative = template_path.trim().trim_start_matches('/');
    let repo_dir = crate::config::app_support_dir()
        .join("hf-templates")
        .join(sanitize_hf_cache_segment(repo_id));
    let cached_path = repo_dir.join(template_relative);

    if cached_path.exists() {
        return Ok(cached_path);
    }

    if let Some(parent) = cached_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| format!("Failed to create {}: {error}", parent.display()))?;
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id.trim(),
        template_relative
    );
    let client = reqwest::Client::builder()
        .user_agent("InferenceBridge/1.0")
        .build()
        .map_err(|error| format!("Failed to create Hugging Face client: {error}"))?;
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|error| format!("Failed to fetch repo chat template from {url}: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Failed to fetch repo chat template from {url}: HTTP {}",
            response.status()
        ));
    }

    let body = response
        .text()
        .await
        .map_err(|error| format!("Failed to read template response from {url}: {error}"))?;
    tokio::fs::write(&cached_path, body)
        .await
        .map_err(|error| format!("Failed to write {}: {error}", cached_path.display()))?;
    Ok(cached_path)
}

async fn resolve_template_selection(
    process: &ProcessConfig,
    model_filename: &str,
    hf_metadata: Option<&HfModelMetadata>,
) -> Result<
    (
        String,
        Option<String>,
        Option<std::path::PathBuf>,
        Option<String>,
        bool,
    ),
    String,
> {
    let requested_mode = process.template_mode.trim().to_lowercase();
    let custom_template = process
        .custom_template_path
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from);

    if let Some(custom_template) = custom_template {
        if custom_template.exists() {
            return Ok((
                "custom".to_string(),
                Some(format!("custom:{}", custom_template.display())),
                Some(custom_template),
                None,
                process.use_jinja || requested_mode == "custom",
            ));
        }
        tracing::warn!(
            model = %model_filename,
            path = %custom_template.display(),
            "Custom template override was configured but the file does not exist; falling back"
        );
    }

    let repo_template_path = hf_metadata
        .and_then(|metadata| metadata.template_path.as_deref())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("chat_template.jinja");

    if requested_mode != "builtin" {
        if let Some(metadata) = hf_metadata {
            if let Some(repo_id) = metadata
                .repo_id
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                if metadata.has_repo_template || metadata.template_path.is_some() {
                    let cached = ensure_cached_hf_template(repo_id, repo_template_path).await?;
                    return Ok((
                        "repo".to_string(),
                        Some(format!("hf:{repo_id}/{repo_template_path}")),
                        Some(cached),
                        None,
                        true,
                    ));
                }
            }
        }
    }

    let template_name = process
        .template_name
        .as_ref()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    Ok((
        "builtin".to_string(),
        template_name
            .as_ref()
            .map(|name| format!("builtin:{name}"))
            .or({
                if process.use_jinja {
                    Some("gguf:embedded-jinja".to_string())
                } else {
                    Some("builtin:fallback".to_string())
                }
            }),
        None,
        template_name,
        process.use_jinja,
    ))
}

fn qwen_profile_needs_jinja(profile: &crate::models::profiles::ModelProfile) -> bool {
    matches!(
        profile.tool_call_format,
        crate::models::profiles::ToolCallFormat::QwenXml
    )
}

fn preview_matches_effective_launch(
    preview: &LaunchPreview,
    model_filename: &str,
    requested_context_size: Option<u32>,
    template_mode: &str,
    template_source: Option<&str>,
    template_name: Option<&str>,
    template_path: Option<&std::path::Path>,
    chat_template_kwargs_json: Option<&str>,
    fit_mode: Option<&str>,
    cache_ram_mb: Option<u32>,
    ctxcp: Option<u32>,
    use_jinja: bool,
    reasoning_mode: Option<&str>,
    extra_args: &[String],
) -> bool {
    let model_ok = names_match(&preview.model_path, model_filename)
        || preview
            .model_path
            .rsplit(std::path::MAIN_SEPARATOR)
            .next()
            .map(|value| names_match(value, model_filename))
            .unwrap_or(false)
        || preview
            .hf_file
            .as_deref()
            .map(|value| names_match(value, model_filename))
            .unwrap_or(false);

    model_ok
        && preview.context_size == requested_context_size
        && preview.template_mode == template_mode
        && preview.template_source.as_deref() == template_source
        && preview.template_name.as_deref() == template_name
        && preview.template_path.as_deref()
            == template_path
                .map(|path| path.to_string_lossy().as_ref().to_string())
                .as_deref()
        && preview.chat_template_kwargs_json.as_deref() == chat_template_kwargs_json
        && preview.fit_mode.as_deref() == fit_mode
        && preview.cache_ram_mb == cache_ram_mb
        && preview.ctxcp == ctxcp
        && preview.use_jinja == use_jinja
        && preview.reasoning_mode.as_deref() == reasoning_mode
        && preview
            .args
            .iter()
            .rev()
            .take(extra_args.len())
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            == extra_args
}

#[cfg(test)]
mod tests {
    use super::resolve_launch_context_size;

    #[test]
    fn passes_through_requested_context() {
        assert_eq!(resolve_launch_context_size(Some(32768)), Some(32768));
    }

    #[test]
    fn none_when_nothing_requested() {
        assert_eq!(resolve_launch_context_size(None), None);
    }

    #[test]
    fn ignores_zero() {
        assert_eq!(resolve_launch_context_size(Some(0)), None);
    }
}

async fn publish_model_load_progress(
    state: &SharedState,
    transition: crate::state::ModelLoadState,
    stage: &str,
    msg: &str,
    progress: f32,
    done: bool,
    error: Option<String>,
) {
    let payload = crate::state::LoadProgress {
        stage: stage.to_string(),
        message: msg.to_string(),
        progress,
        done,
        error: error.clone(),
    };

    let app_handle = {
        let mut s = state.write().await;
        s.model_load_state = if done {
            done_model_load_state(&transition, error.as_deref())
        } else {
            transition.clone()
        };
        s.model_load_progress = Some(payload.clone());
        s.app_handle.clone()
    };

    emit_load_progress_payload(&app_handle, &payload);
}

fn model_load_state_label(state: &crate::state::ModelLoadState) -> String {
    match state {
        crate::state::ModelLoadState::Idle => "Idle".to_string(),
        crate::state::ModelLoadState::Loading => "Loading".to_string(),
        crate::state::ModelLoadState::Swapping => "Swapping".to_string(),
        crate::state::ModelLoadState::Unloading => "Unloading".to_string(),
        crate::state::ModelLoadState::Loaded => "Loaded".to_string(),
        crate::state::ModelLoadState::Error(_) => "Error".to_string(),
    }
}

async fn store_launch_preview(state: &SharedState, preview: LaunchPreview) {
    let mut s = state.write().await;
    s.last_launch_preview = Some(preview);
}

fn empty_context_status(total_tokens: u32) -> crate::context::tracker::ContextStatus {
    crate::context::tracker::ContextStatus {
        total_tokens,
        used_tokens: 0,
        fill_ratio: 0.0,
        pinned_tokens: 0,
        rolling_tokens: 0,
        compressed_tokens: 0,
        last_compaction_action: None,
    }
}

fn clear_runtime_after_backend_exit(
    state: &mut crate::state::AppState,
    error_message: Option<String>,
) {
    if let Some(loaded) = state.loaded_model.take() {
        state.previous_model = Some(loaded);
    }
    state.model_stats = None;
    state.last_context_status = Some(crate::context::tracker::ContextStatus::empty());

    if let Some(error_message) = error_message {
        state.model_load_state = crate::state::ModelLoadState::Error(error_message.clone());
        state.model_load_progress = Some(crate::state::LoadProgress {
            stage: "error".to_string(),
            message: error_message.clone(),
            progress: 0.0,
            done: true,
            error: Some(error_message),
        });
    } else {
        state.model_load_state = crate::state::ModelLoadState::Idle;
        state.model_load_progress = None;
    }
}

fn effective_profile_info_from_state(
    state: &crate::state::AppState,
    requested_model: Option<&str>,
) -> Result<EffectiveProfileInfo, String> {
    let resolved_model = requested_model
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| state.loaded_model.clone())
        .or_else(|| {
            state
                .model_registry
                .list()
                .first()
                .map(|model| model.filename.clone())
        })
        .ok_or_else(|| "No model is available to resolve an effective profile".to_string())?;

    Ok(EffectiveProfileInfo {
        requested_model: requested_model
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        override_entry: effective_override(&resolved_model),
        profile: detect_effective_profile(&resolved_model),
        resolved_model: Some(resolved_model),
    })
}

pub fn get_effective_profile_for_shared(
    state: &crate::state::AppState,
    model_name: Option<&str>,
) -> Result<EffectiveProfileInfo, String> {
    effective_profile_info_from_state(state, model_name)
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeLoadOverrides {
    pub hf_repo: Option<String>,
    pub hf_file: Option<String>,
    pub fit_mode: Option<String>,
    pub cache_ram_mb: Option<u32>,
    pub ctxcp: Option<u32>,
    pub use_jinja: Option<bool>,
    pub reasoning_mode: Option<String>,
    pub template_mode: Option<String>,
    pub template_name: Option<String>,
    pub custom_template_path: Option<String>,
    pub chat_template_kwargs_json: Option<String>,
    pub extra_args: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeLoadRequest {
    #[serde(default)]
    pub context_size: Option<u32>,
    #[serde(default)]
    pub hf_repo: Option<String>,
    #[serde(default)]
    pub hf_file: Option<String>,
    #[serde(default)]
    pub fit_mode: Option<String>,
    #[serde(default)]
    pub cache_ram_mb: Option<u32>,
    #[serde(default)]
    pub ctxcp: Option<u32>,
    #[serde(default)]
    pub use_jinja: Option<bool>,
    #[serde(default)]
    pub reasoning_mode: Option<String>,
    #[serde(default)]
    pub template_mode: Option<String>,
    #[serde(default)]
    pub template_name: Option<String>,
    #[serde(default)]
    pub custom_template_path: Option<String>,
    #[serde(default)]
    pub chat_template_kwargs_json: Option<String>,
    #[serde(default)]
    pub extra_args: Option<Vec<String>>,
}

impl RuntimeLoadRequest {
    pub fn normalized_context_size(&self) -> Option<u32> {
        normalize_requested_context_size(self.context_size)
    }

    pub fn into_overrides(self) -> RuntimeLoadOverrides {
        RuntimeLoadOverrides {
            hf_repo: self.hf_repo,
            hf_file: self.hf_file,
            fit_mode: self.fit_mode,
            cache_ram_mb: self.cache_ram_mb,
            ctxcp: self.ctxcp,
            use_jinja: self.use_jinja,
            reasoning_mode: self.reasoning_mode,
            template_mode: self.template_mode,
            template_name: self.template_name,
            custom_template_path: self.custom_template_path,
            chat_template_kwargs_json: self.chat_template_kwargs_json,
            extra_args: self.extra_args,
        }
    }
}

/// Core backend model loading logic, used by both the REST API and headless mode.
///
/// Emits `model-load-progress` Tauri events when an `app_handle` is stored in
/// AppState (GUI mode). In headless/API-only mode the handle is `None` and the
/// function is silent toward the frontend.
pub async fn backend_load_model_with_overrides(
    state: SharedState,
    model_name: String,
    context_size: Option<u32>,
    overrides: RuntimeLoadOverrides,
) -> Result<String, String> {
    // Cancel any active inference before starting a new load.
    {
        let mut s = state.write().await;
        if s.active_generation.is_some() {
            s.generation_cancel.cancel();
            s.cumulative_metrics.total_cancellations += 1;
            tracing::info!("Cancelled active inference for new model load");
        }
        s.cumulative_metrics.total_model_loads += 1;
        // Fresh cancellation token for this load.
        s.model_load_cancel = tokio_util::sync::CancellationToken::new();
    }

    let load_mutex = {
        let s = state.read().await;
        s.model_load_mutex.clone()
    };
    let _load_guard = load_mutex.lock().await;

    let transition = {
        let s = state.read().await;
        load_transition_for_request(s.loaded_model.as_deref(), &model_name)
    };

    // Grab the app handle once (no lock held after this point).
    let app_handle = {
        let s = state.read().await;
        s.app_handle.clone()
    };

    publish_model_load_progress(
        &state,
        transition.clone(),
        "resolving",
        "Resolving model...",
        0.0,
        false,
        None,
    )
    .await;

    // Phase 1: Resolve model info (brief lock) + claim loading generation
    let (config, model_filename, my_generation, launch_preview, reused_existing) = {
        let model = {
            let s = state.read().await;
            s.model_registry.find_by_name(&model_name).cloned()
        };

        let model = if let Some(model) = model {
            model
        } else {
            let scan_dirs = {
                let s = state.read().await;
                s.config.models.scan_dirs.clone()
            };
            let scanned = tokio::task::spawn_blocking(move || scanner::scan_all(&scan_dirs))
                .await
                .map_err(|e| format!("Failed to rescan models: {e}"))?;

            let mut s = state.write().await;
            s.model_registry.update(scanned);
            s.model_registry
                .find_by_name(&model_name)
                .cloned()
                .ok_or_else(|| {
                    let msg = format!("Model not found: {model_name}");
                    let payload = crate::state::LoadProgress {
                        stage: "error".to_string(),
                        message: msg.clone(),
                        progress: 0.0,
                        done: true,
                        error: Some(msg.clone()),
                    };
                    emit_load_progress_payload(&app_handle, &payload);
                    msg
                })?
        };

        let mut s = state.write().await;
        let hf_metadata = model.hf_metadata.clone();
        let ctx = resolve_launch_context_size(context_size);
        let mut process_config = s.config.process.clone();
        if let Some(fit_mode) = overrides.fit_mode.as_ref() {
            process_config.fit_mode = fit_mode.clone();
        }
        if let Some(cache_ram_mb) = overrides.cache_ram_mb {
            process_config.cache_ram_mb = Some(cache_ram_mb);
        }
        if let Some(ctxcp) = overrides.ctxcp {
            process_config.ctxcp = Some(ctxcp);
        }
        if let Some(use_jinja) = overrides.use_jinja {
            process_config.use_jinja = use_jinja;
        }
        if let Some(reasoning_mode) = overrides.reasoning_mode.as_ref() {
            process_config.reasoning_mode = reasoning_mode.clone();
        }
        if let Some(template_mode) = overrides.template_mode.as_ref() {
            process_config.template_mode = template_mode.clone();
        }
        if let Some(template_name) = overrides.template_name.as_ref() {
            process_config.template_name = Some(template_name.clone());
        }
        if let Some(custom_template_path) = overrides.custom_template_path.as_ref() {
            process_config.custom_template_path = Some(custom_template_path.clone());
        }
        if let Some(chat_template_kwargs_json) = overrides.chat_template_kwargs_json.as_ref() {
            process_config.chat_template_kwargs_json = Some(chat_template_kwargs_json.clone());
        }
        if let Some(extra_args) = overrides.extra_args.as_ref() {
            process_config.extra_args = extra_args.clone();
        }

        let effective_profile = detect_effective_profile(&model.filename);
        if qwen_profile_needs_jinja(&effective_profile) && !process_config.use_jinja {
            tracing::warn!(
                model = %model.filename,
                "Qwen tool-call profile requires llama.cpp Jinja chat-template mode; enabling --jinja for this launch"
            );
            process_config.use_jinja = true;
        }

        let (template_mode, template_source, template_file, template_name, use_jinja) =
            resolve_template_selection(&process_config, &model.filename, hf_metadata.as_ref())
                .await?;

        // If ctx is None (no one specified), use 0 as placeholder — will be
        // updated from /slots or /props after health check succeeds.
        let provisional_ctx = ctx.unwrap_or(0);
        s.last_context_status = Some(empty_context_status(provisional_ctx));
        s.model_stats = Some(crate::state::ModelStats {
            model: model.filename.clone(),
            context_size: provisional_ctx,
            tokens_per_sec: 0.0,
            memory_mb: 0,
        });

        let config = LaunchConfig {
            model_path: model.path.clone(),
            hf_repo: overrides.hf_repo.clone().or_else(|| {
                hf_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.repo_id.clone())
            }),
            hf_file: overrides.hf_file.clone().or_else(|| {
                hf_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.file.clone())
            }),
            context_size: ctx,
            gpu_layers: s.config.process.gpu_layers,
            threads: s.config.process.threads,
            threads_batch: s.config.process.threads_batch,
            port: 0, // auto-assign ephemeral port at launch
            backend_preference: s.config.process.backend_preference.clone(),
            batch_size: s.config.process.batch_size,
            ubatch_size: s.config.process.ubatch_size,
            flash_attn: s.config.process.flash_attn,
            use_mmap: s.config.process.use_mmap,
            use_mlock: s.config.process.use_mlock,
            cont_batching: s.config.process.cont_batching,
            parallel_slots: s.config.process.parallel_slots,
            main_gpu: s.config.process.main_gpu,
            defrag_thold: s.config.process.defrag_thold,
            rope_freq_scale: s.config.process.rope_freq_scale,
            fit_mode: Some(process_config.fit_mode.clone())
                .filter(|value| !value.trim().is_empty()),
            cache_ram_mb: process_config.cache_ram_mb,
            ctxcp: process_config.ctxcp,
            use_jinja,
            reasoning_mode: Some(process_config.reasoning_mode.clone())
                .filter(|value| !value.trim().is_empty()),
            template_mode: template_mode.clone(),
            template_source: template_source.clone(),
            template_file: template_file.clone(),
            template_name: template_name.clone(),
            chat_template_kwargs_json: process_config
                .chat_template_kwargs_json
                .clone()
                .filter(|value| !value.trim().is_empty()),
            extra_args: process_config.extra_args.clone(),
            cache_type_k: s.config.process.cache_type_k.clone(),
            cache_type_v: s.config.process.cache_type_v.clone(),
            kv_unified: s.config.process.kv_unified,
            no_warmup: s.config.process.no_warmup,
            ctx_shift: s.config.process.ctx_shift,
            tensor_split: s.config.process.tensor_split.clone(),
            draft_model_path: s.config.process.draft_model_path.clone(),
            draft_max_tokens: s.config.process.draft_max_tokens,
            draft_min_tokens: s.config.process.draft_min_tokens,
            draft_p_min: s.config.process.draft_p_min,
        };

        LlamaProcess::validate_launch_config(&config).map_err(|e| {
            let msg = format!("Invalid launch configuration: {e}");
            let payload = crate::state::LoadProgress {
                stage: "error".to_string(),
                message: msg.clone(),
                progress: 0.0,
                done: true,
                error: Some(msg.clone()),
            };
            emit_load_progress_payload(&app_handle, &payload);
            msg
        })?;

        let preview = s.process.build_args_preview(&config).map_err(|e| {
            let msg = format!("Could not build launch preview: {e}");
            let payload = crate::state::LoadProgress {
                stage: "error".to_string(),
                message: msg.clone(),
                progress: 0.0,
                done: true,
                error: Some(msg.clone()),
            };
            emit_load_progress_payload(&app_handle, &payload);
            msg
        })?;

        let reuse_existing = if let (Some(loaded), Some(last_preview)) =
            (s.loaded_model.clone(), s.last_launch_preview.as_ref())
        {
            if names_match(&loaded, &model.filename)
                && preview_matches_effective_launch(
                    last_preview,
                    &model.filename,
                    ctx,
                    &template_mode,
                    template_source.as_deref(),
                    template_name.as_deref(),
                    template_file.as_deref(),
                    config.chat_template_kwargs_json.as_deref(),
                    config.fit_mode.as_deref(),
                    config.cache_ram_mb,
                    config.ctxcp,
                    config.use_jinja,
                    config.reasoning_mode.as_deref(),
                    &config.extra_args,
                )
            {
                tracing::info!(
                    model = %model.filename,
                    context_size = ?ctx,
                    template_mode = %template_mode,
                    "Model already loaded with matching effective launch config"
                );
                Some((loaded, last_preview.clone()))
            } else {
                None
            }
        } else {
            None
        };

        if let Some((loaded, preview)) = reuse_existing {
            (config, loaded, s.loading_generation, preview, true)
        } else {
            // Bump generation so older in-flight loads won't overwrite us
            s.loading_generation += 1;
            let gen = s.loading_generation;
            tracing::info!(
                model = %model.filename,
                generation = gen,
                ctx_size = ctx,
                "Phase 1: Resolved model (API/backend)"
            );

            (config, model.filename.clone(), gen, preview, false)
        }
    }; // write lock released

    if reused_existing {
        store_launch_preview(&state, launch_preview.clone()).await;
        return Ok(model_filename);
    }

    store_launch_preview(&state, launch_preview.clone()).await;

    let size_info = match config.context_size {
        Some(ctx) => format!("{} (ctx: {})", model_filename, ctx),
        None => format!("{} (ctx: model-default)", model_filename),
    };

    // Phase 2: Launch process (brief write lock, then released)
    //
    publish_model_load_progress(
        &state,
        transition.clone(),
        "launching",
        &format!("Launching llama-server for {}...", model_filename),
        0.05,
        false,
        None,
    )
    .await;
    {
        let mut s = state.write().await;
        s.process.launch(config).await.map_err(|e| {
            let msg = format!("Failed to launch llama-server: {e}");
            emit_load_progress(&app_handle, "error", &msg, 0.0, true, Some(msg.clone()));
            msg
        })?;
    } // write lock released

    // Read the actual port assigned by the OS (resolved from 0 during launch).
    let backend_port = {
        let s = state.read().await;
        s.process.port()
    };
    tracing::info!(backend_port, "llama-server launched on auto-assigned port");

    emit_load_progress(
        &app_handle,
        "loading",
        &format!("Loading {} into memory...", model_filename),
        0.1,
        false,
        None,
    );
    let (load_timeout_secs, health_poll_ms, load_cancel) = {
        let s = state.read().await;
        (
            s.config.process.model_load_timeout_secs,
            s.config.process.health_poll_interval_ms,
            s.model_load_cancel.clone(),
        )
    };
    tracing::info!(
        port = backend_port,
        timeout_secs = load_timeout_secs,
        "Phase 3: Waiting for llama-server health check..."
    );

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            tracing::warn!(%error, "Falling back to default HTTP client for backend health check");
            reqwest::Client::new()
        }
    };
    let health_url = format!("http://127.0.0.1:{}/health", backend_port);
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(load_timeout_secs);
    let mut attempt = 0u32;

    loop {
        if start.elapsed() > timeout {
            let msg = format!("llama-server did not become healthy within {:?}", timeout);
            tracing::error!("{msg}");
            emit_load_progress(&app_handle, "error", &msg, 0.0, true, Some(msg.clone()));
            let mut s = state.write().await;
            let _ = s.process.shutdown().await;
            clear_runtime_after_backend_exit(&mut s, Some(msg.clone()));
            return Err(msg);
        }

        if load_cancel.is_cancelled() {
            let msg = "Model load cancelled by new request".to_string();
            tracing::info!("{msg}");
            let mut s = state.write().await;
            let _ = s.process.shutdown().await;
            clear_runtime_after_backend_exit(&mut s, Some(msg.clone()));
            return Err(msg);
        }

        // Check if process crashed — every ~13 iterations (~2 seconds).
        let process_exited = if attempt % 13 == 0 {
            let mut s = state.write().await;
            s.process.poll_exited()
        } else {
            false
        };
        if process_exited {
            // Give the background stderr-reader task a moment to drain — outside the lock.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let mut s = state.write().await;
            let stderr = s.process.last_stderr().await;
            let last_lines: String = stderr
                .iter()
                .rev()
                .take(10)
                .rev()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            let msg = format!(
                "llama-server crashed on startup.\n{}",
                if last_lines.is_empty() {
                    "No stderr output captured.".to_string()
                } else {
                    last_lines
                }
            );
            tracing::error!("{msg}");
            emit_load_progress(&app_handle, "error", &msg, 0.0, true, Some(msg.clone()));
            clear_runtime_after_backend_exit(&mut s, Some(msg.clone()));
            return Err(msg);
        }

        match client.get(&health_url).send().await {
            Ok(resp) => {
                let status_code = resp.status();
                let body = resp.text().await.unwrap_or_default();

                if status_code.is_success() {
                    if body.contains("ok") || body.contains("\"status\":\"ok\"") {
                        tracing::info!("llama-server healthy after {}s", start.elapsed().as_secs());
                        break;
                    }
                    if body.contains("loading") {
                        let elapsed = start.elapsed().as_secs();
                        let progress = (0.1 + (elapsed as f32 * 0.01)).min(0.9);
                        emit_load_progress(
                            &app_handle,
                            "loading",
                            &format!("Loading model into GPU memory... ({}s)", elapsed),
                            progress,
                            false,
                            None,
                        );
                    } else {
                        break; // 200 with unknown body — assume ready
                    }
                } else if status_code.as_u16() == 503 {
                    let elapsed = start.elapsed().as_secs();
                    let progress = (0.1 + (elapsed as f32 * 0.01)).min(0.9);
                    emit_load_progress(
                        &app_handle,
                        "loading",
                        &format!("Loading model weights... ({}s)", elapsed),
                        progress,
                        false,
                        None,
                    );
                } else {
                    let elapsed = start.elapsed().as_secs();
                    emit_load_progress(
                        &app_handle,
                        "starting",
                        &format!(
                            "Waiting for llama-server (HTTP {})... ({}s)",
                            status_code, elapsed
                        ),
                        0.08,
                        false,
                        None,
                    );
                }
            }
            Err(_) => {
                let elapsed = start.elapsed().as_secs();
                let progress = (0.02 + (elapsed as f32 * 0.005)).min(0.09);
                emit_load_progress(
                    &app_handle,
                    "starting",
                    &format!("Waiting for llama-server to respond... ({}s)", elapsed),
                    progress,
                    false,
                    None,
                );
            }
        }

        attempt += 1;
        if attempt < 5 || attempt % 10 == 0 {
            tracing::debug!(attempt, "Health check attempt");
        }
        tokio::time::sleep(std::time::Duration::from_millis(health_poll_ms)).await;
    }

    // Phase 4: Mark as loaded (brief write lock)
    {
        let mut s = state.write().await;
        if s.loading_generation != my_generation {
            tracing::warn!(
                model = %model_filename,
                our_gen = my_generation,
                current_gen = s.loading_generation,
                "Stale load — a newer swap is in progress, discarding"
            );
            return Ok(format!("Superseded by newer swap (gen {})", my_generation));
        }
        if let Some(prev) = s.loaded_model.take() {
            if prev != model_filename {
                s.previous_model = Some(prev);
            }
        }
        s.loaded_model = Some(model_filename.clone());
        s.process.set_state_running();
        s.last_known_good_config = Some(launch_preview);
        s.last_startup_duration_ms = Some(start.elapsed().as_millis() as u64);
    }

    crate::context::tracker::reset_slots_warning();

    // Phase 4b: Discover actual context size from the running server.
    // When we didn't pass --ctx-size, llama-server picks the model's native
    // window.  Query /slots (or /props) to learn the real n_ctx.
    {
        let llama_client = crate::engine::client::LlamaClient::new(backend_port);
        let actual_ctx = match llama_client.get_slots().await {
            Ok(slots) if !slots.is_empty() => Some(slots[0].n_ctx),
            _ => match llama_client.get_props().await {
                Ok(props) => props
                    .default_generation_settings
                    .and_then(|s| s.n_ctx)
                    .filter(|v| *v > 0),
                Err(_) => None,
            },
        };

        if let Some(real_ctx) = actual_ctx {
            let mut s = state.write().await;
            s.last_context_status = Some(empty_context_status(real_ctx));
            if let Some(stats) = s.model_stats.as_mut() {
                stats.context_size = real_ctx;
            }
            // Sync the real context back into last_launch_preview so that
            // resolve_loaded_model can correctly detect context mismatches on
            // subsequent API requests (e.g. if no --ctx-size was explicitly
            // requested, preview.context_size was None and any explicit-ctx
            // request would always trigger an unnecessary reload).
            if let Some(preview) = s.last_launch_preview.as_mut() {
                preview.context_size = Some(real_ctx);
            }
            tracing::info!(
                real_ctx,
                "Discovered actual context size from running server"
            );
        }
    }

    let result = format!("Loaded {size_info}");
    publish_model_load_progress(
        &state,
        transition.clone(),
        "ready",
        &result,
        1.0,
        true,
        None,
    )
    .await;
    tracing::info!("{result}");
    Ok(result)
}

pub async fn backend_load_model(
    state: SharedState,
    model_name: String,
    context_size: Option<u32>,
) -> Result<String, String> {
    backend_load_model_with_overrides(
        state,
        model_name,
        context_size,
        RuntimeLoadOverrides::default(),
    )
    .await
}
// Tauri commands for model management.

use crate::config::ProcessConfig;
use crate::engine::download;
use crate::engine::process::{LaunchConfig, LaunchPreview, LlamaProcess};
use crate::models::overrides::{detect_effective_profile, effective_override, HfModelMetadata};
use crate::models::scanner;
use crate::state::{
    EffectiveProfileInfo, GenerationRequest, LoadProgress, RuntimePerformanceMetrics, SharedState,
};
fn command_no_window(program: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}

#[tauri::command]
pub async fn list_models(state: tauri::State<'_, SharedState>) -> Result<Vec<ModelInfo>, String> {
    if let Some(models) = list_active_external_provider_models(state.inner().clone()).await {
        return Ok(models);
    }

    let s = state.read().await;
    let loaded_model = s.loaded_model.clone();
    let loaded_context = s.model_stats.as_ref().map(|stats| stats.context_size);
    let last_launch_preview = s.last_launch_preview.clone();
    Ok(s.model_registry
        .list()
        .iter()
        .map(|m| {
            use crate::models::profiles::{ThinkTagStyle, ToolCallFormat};
            let supports_tools = !matches!(m.profile.tool_call_format, ToolCallFormat::NativeApi)
                || m.profile.supports_parallel_tools;
            let supports_reasoning = !matches!(m.profile.think_tag_style, ThinkTagStyle::None);
            let is_loaded = loaded_model
                .as_deref()
                .map(|loaded| names_match(loaded, &m.filename))
                .unwrap_or(false);
            let vision_runtime_ready = is_loaded
                && last_launch_preview
                    .as_ref()
                    .and_then(|preview| preview.mmproj_path.as_ref())
                    .is_some();
            ModelInfo {
                filename: m.filename.clone(),
                path: m.path.to_string_lossy().to_string(),
                size_gb: m.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                family: m.profile.family.to_string(),
                supports_tools,
                supports_reasoning,
                supports_vision: m.profile.supports_vision,
                context_window: if is_loaded {
                    loaded_context
                } else {
                    m.profile.default_context_window
                },
                // GGUF context_length is the ground-truth training context;
                // fall back to the profile hardcode when parsing failed.
                max_context_window: m
                    .gguf_meta
                    .as_ref()
                    .and_then(|g| g.context_length)
                    .or(m.profile.max_context_window),
                max_output_tokens: m.profile.default_max_output_tokens,
                default_temperature: m.profile.default_temperature,
                default_top_p: m.profile.default_top_p,
                default_top_k: m.profile.default_top_k,
                default_min_p: m.profile.default_min_p,
                default_presence_penalty: m.profile.default_presence_penalty,
                quant: extract_quant(&m.filename),
                tool_call_format: format!("{:?}", m.profile.tool_call_format),
                think_tag_style: format!("{:?}", m.profile.think_tag_style),
                hf_repo: m
                    .hf_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.repo_id.clone()),
                hf_file: m
                    .hf_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.file.clone()),
                template_mode: if is_loaded {
                    last_launch_preview
                        .as_ref()
                        .map(|preview| preview.template_mode.clone())
                } else {
                    None
                },
                template_source: if is_loaded {
                    last_launch_preview
                        .as_ref()
                        .and_then(|preview| preview.template_source.clone())
                } else {
                    None
                },
                vision_runtime_ready,
                vision_status: if !m.profile.supports_vision {
                    "Not capable".to_string()
                } else if vision_runtime_ready {
                    "Vision Ready".to_string()
                } else if is_loaded {
                    "mmproj Missing".to_string()
                } else {
                    "Vision Capable".to_string()
                },
                provider_type: "managed_llamacpp".to_string(),
                provider_name: "Managed llama.cpp".to_string(),
                provider_base_url: None,
                provider_managed: true,
                n_layers: m.gguf_meta.as_ref().and_then(|g| g.n_layers),
                n_kv_heads: m.gguf_meta.as_ref().and_then(|g| g.n_kv_heads),
                head_dim: m.gguf_meta.as_ref().and_then(|g| g.head_dim()),
                gguf_architecture: m.gguf_meta.as_ref().and_then(|g| g.architecture.clone()),
            }
        })
        .collect())
}

#[derive(serde::Deserialize)]
struct ProviderModelsResponse {
    #[serde(default)]
    data: Vec<ProviderModelObject>,
}

#[derive(serde::Deserialize)]
struct ProviderModelObject {
    id: String,
    #[serde(default)]
    max_context_length: Option<u32>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    context_window: Option<u32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
}

async fn list_active_external_provider_models(state: SharedState) -> Option<Vec<ModelInfo>> {
    let provider = crate::api::upstream::active_openai_provider(&state).await?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;
    let mut request = client.get(format!(
        "{}/models",
        provider.base_url.trim_end_matches('/')
    ));
    if let Some(api_key) = provider
        .api_key
        .clone()
        .filter(|value| !value.trim().is_empty())
    {
        request = request.bearer_auth(api_key);
    }
    let response = request.send().await.ok()?;
    if !response.status().is_success() {
        return Some(Vec::new());
    }
    let upstream: ProviderModelsResponse = response.json().await.ok()?;
    Some(
        upstream
            .data
            .into_iter()
            .map(|model| {
                let context = model
                    .max_context_length
                    .or(model.context_length)
                    .or(model.context_window);
                ModelInfo {
                    filename: model.id,
                    path: provider.base_url.clone(),
                    size_gb: 0.0,
                    family: provider.name.clone(),
                    supports_tools: false,
                    supports_reasoning: false,
                    supports_vision: false,
                    context_window: context,
                    max_context_window: context,
                    max_output_tokens: model.max_output_tokens,
                    default_temperature: None,
                    default_top_p: None,
                    default_top_k: None,
                    default_min_p: None,
                    default_presence_penalty: None,
                    quant: None,
                    tool_call_format: "ProviderNative".to_string(),
                    think_tag_style: "None".to_string(),
                    hf_repo: None,
                    hf_file: None,
                    template_mode: None,
                    template_source: Some(provider.name.clone()),
                    vision_runtime_ready: false,
                    vision_status: "Provider managed".to_string(),
                    provider_type: "lm_studio".to_string(),
                    provider_name: provider.name.clone(),
                    provider_base_url: Some(provider.base_url.clone()),
                    provider_managed: false,
                    n_layers: None,
                    n_kv_heads: None,
                    head_dim: None,
                    gguf_architecture: None,
                }
            })
            .collect(),
    )
}

#[tauri::command]
pub async fn scan_models(state: tauri::State<'_, SharedState>) -> Result<usize, String> {
    let mut s = state.write().await;
    let models = scanner::scan_all(&s.config.models.scan_dirs);
    let count = models.len();
    s.model_registry.update(models);
    Ok(count)
}

#[tauri::command]
pub async fn load_model(
    _app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
    model_name: String,
    options: Option<RuntimeLoadRequest>,
) -> Result<String, String> {
    let options = options.unwrap_or_default();
    backend_load_model_with_overrides(
        state.inner().clone(),
        model_name,
        options.normalized_context_size(),
        options.into_overrides(),
    )
    .await
}

pub async fn backend_unload_model(state: SharedState) -> Result<String, String> {
    // Serialize with load mutex to prevent race with concurrent loads.
    let load_mutex = {
        let s = state.read().await;
        s.model_load_mutex.clone()
    };
    let _load_guard = load_mutex.lock().await;

    // Cancel active inference.
    {
        let mut s = state.write().await;
        if s.active_generation.is_some() {
            s.generation_cancel.cancel();
            tracing::info!("Cancelled active inference for model unload");
        }
        s.cumulative_metrics.total_model_unloads += 1;
    }

    let unloaded_model = {
        let s = state.read().await;
        s.loaded_model.clone()
    };

    if unloaded_model.is_none() {
        let mut s = state.write().await;
        clear_runtime_after_backend_exit(&mut s, None);
        s.model_load_progress = None;
        s.last_context_status = Some(crate::context::tracker::ContextStatus::empty());
        return Ok("No model was loaded.".to_string());
    }

    publish_model_load_progress(
        &state,
        crate::state::ModelLoadState::Unloading,
        "unloading",
        "Unloading model...",
        0.0,
        false,
        None,
    )
    .await;

    {
        let mut s = state.write().await;
        s.generation_cancel.cancel();
        s.active_generation = None;
        s.process.shutdown().await.map_err(|error| {
            let message = format!("Failed to shut down llama-server: {error}");
            clear_runtime_after_backend_exit(&mut s, Some(message.clone()));
            message
        })?;
        clear_runtime_after_backend_exit(&mut s, None);
        s.model_load_progress = None;
    }

    crate::context::tracker::reset_slots_warning();

    let result = format!(
        "Unloaded {}",
        unloaded_model.unwrap_or_else(|| "model".to_string())
    );
    publish_model_load_progress(
        &state,
        crate::state::ModelLoadState::Unloading,
        "ready",
        &result,
        1.0,
        true,
        None,
    )
    .await;
    Ok(result)
}

#[tauri::command]
pub async fn unload_model(state: tauri::State<'_, SharedState>) -> Result<String, String> {
    backend_unload_model(state.inner().clone()).await
}

#[tauri::command]
pub async fn get_process_status(
    state: tauri::State<'_, SharedState>,
) -> Result<ProcessStatusInfo, String> {
    collect_process_status(state.inner().clone()).await
}

pub async fn collect_process_status(state: SharedState) -> Result<ProcessStatusInfo, String> {
    {
        let mut s = state.write().await;
        if s.process.check_crashed().await {
            clear_runtime_after_backend_exit(
                &mut s,
                Some("llama-server exited unexpectedly.".to_string()),
            );
        }
    }

    // Extract everything we need from the lock, then drop it before any I/O
    let (
        process_state,
        is_running,
        _port,
        loaded_model,
        previous_model,
        crash_count,
        configured_server_path,
        detected_arc,
        api_state,
        api_error,
        api_url,
        api_port,
        last_launch_preview,
        startup_duration_ms,
        parallel_slots,
        scheduler_snapshot,
        model_load_state,
        model_load_progress,
        active_generation,
        last_generation_metrics,
        live_stream,
    ) = {
        let s = state.read().await;
        let process_state = s.process.state();
        let is_running = process_state != crate::engine::process::ProcessState::Idle;
        (
            process_state,
            is_running,
            s.process.port(),
            s.loaded_model.clone(),
            s.previous_model.clone(),
            s.process.crash_count(),
            s.config.process.llama_server_path.clone(),
            s.process.detected_backend(),
            format!("{:?}", s.api_server_state),
            s.api_server_error.clone(),
            crate::api::server::reachable_api_url(&s.config.server.host, s.config.server.port),
            s.config.server.port,
            s.last_launch_preview.clone(),
            s.last_startup_duration_ms,
            Some(s.config.process.parallel_slots),
            s.request_scheduler.snapshot(),
            s.model_load_state.clone(),
            s.model_load_progress.clone(),
            s.active_generation.clone(),
            s.last_generation_metrics.clone(),
            s.live_stream.clone(),
        )
    }; // read lock released before any async I/O

    let server_path = last_launch_preview
        .as_ref()
        .and_then(|preview| {
            (!preview.server_path.trim().is_empty())
                .then(|| std::path::PathBuf::from(preview.server_path.trim()))
        })
        .or_else(|| {
            let explicit = configured_server_path.trim();
            (!explicit.is_empty()).then(|| std::path::PathBuf::from(explicit))
        })
        .or_else(|| {
            let managed =
                crate::engine::process::LlamaProcess::managed_binary_dir().join("llama-server.exe");
            managed.exists().then_some(managed)
        });

    let effective_loaded_model = if loaded_model.is_some() {
        loaded_model.clone()
    } else if is_running {
        last_launch_preview.as_ref().and_then(|preview| {
            std::path::Path::new(&preview.model_path)
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
        })
    } else {
        None
    };

    let state_str = format!("{process_state:?}");
    let mut effective_model_load_state = model_load_state.clone();
    let mut effective_model_load_progress = model_load_progress.clone();

    if matches!(process_state, crate::engine::process::ProcessState::Running)
        && effective_loaded_model.is_some()
    {
        effective_model_load_state = crate::state::ModelLoadState::Loaded;
        effective_model_load_progress = None;
    }

    let model_load_state_label = model_load_state_label(&effective_model_load_state);

    let transition_active = matches!(
        process_state,
        crate::engine::process::ProcessState::Starting
            | crate::engine::process::ProcessState::Stopping
    ) || matches!(
        effective_model_load_state,
        crate::state::ModelLoadState::Loading
            | crate::state::ModelLoadState::Swapping
            | crate::state::ModelLoadState::Unloading
    ) || effective_model_load_progress
        .as_ref()
        .map(|progress| !progress.done)
        .unwrap_or(false);

    let api_reachable = probe_public_api(&api_url).await;

    if api_reachable && (api_state != "Running" || api_error.is_some()) {
        let mut s = state.write().await;
        s.api_server_state = crate::state::ApiServerState::Running;
        s.api_server_error = None;
    }

    // Get backend from live detection (parsed from server stderr on startup)
    let backend = if is_running {
        let guard = detected_arc.lock().await;
        Some(
            guard
                .clone()
                .unwrap_or_else(|| detect_backend_from_path(server_path.as_deref())),
        )
    } else {
        None
    };

    // Try managed binary version file first, then query live server /props
    let server_version = download::current_version();

    let api_port_owner = if !api_reachable && !transition_active {
        detect_api_port_owner(api_port)
    } else {
        None
    };
    let api_owned_by_self = api_port_owner
        .as_ref()
        .map(|owner| owner.kind == "self")
        .unwrap_or(false);

    let effective_api_state = if api_reachable {
        "Running".to_string()
    } else if transition_active || api_owned_by_self {
        "Starting".to_string()
    } else if api_state == "Running" {
        "Error".to_string()
    } else {
        api_state.clone()
    };

    let effective_api_error = if api_reachable || transition_active || api_owned_by_self {
        None
    } else if effective_api_state == "Error" {
        if let Some(owner) = api_port_owner.as_ref() {
            if owner.kind == "self" {
                None
            } else {
                api_error.clone().or_else(|| {
                    Some(format!(
                        "The public API is not currently reachable on {api_url}. Another process appears to be holding port {api_port}."
                    ))
                })
            }
        } else {
            Some(format!(
                "The public API is not currently reachable on {api_url}. No active listener is holding port {api_port}. Retry API to start it again."
            ))
        }
    } else {
        api_error.clone()
    };

    let slot_count = if is_running {
        last_launch_preview
            .as_ref()
            .map(|preview| preview.parallel_slots)
            .filter(|slots| *slots > 0)
    } else {
        None
    };

    Ok(ProcessStatusInfo {
        state: state_str,
        model: effective_loaded_model,
        previous_model,
        crash_count,
        server_version,
        server_path: server_path.map(|p| p.to_string_lossy().to_string()),
        backend,
        api_state: effective_api_state,
        api_error: effective_api_error,
        api_url,
        api_reachable,
        api_port_owner,
        startup_duration_ms,
        parallel_slots,
        slot_count,
        active_requests: scheduler_snapshot.active,
        queued_requests: scheduler_snapshot.queued,
        scheduler_limit: Some(scheduler_snapshot.limit),
        last_launch_preview,
        model_load_state: model_load_state_label,
        model_load_progress: effective_model_load_progress,
        active_generation,
        last_generation_metrics,
        live_stream,
    })
}

async fn probe_public_api(api_url: &str) -> bool {
    let health_url = format!("{}/health", api_url.trim_end_matches('/'));
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(1200))
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };

    match client.get(&health_url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// Detect backend by checking the binary's directory for CUDA/Vulkan DLLs.
fn detect_backend_from_path(path: Option<&std::path::Path>) -> String {
    let Some(path) = path else {
        return "Unknown".to_string();
    };
    let dir = path.parent().unwrap_or(path);

    // Check for CUDA DLLs next to the binary
    let cuda_indicators = [
        "ggml-cuda.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudart64_12.dll",
    ];
    for dll in &cuda_indicators {
        if dir.join(dll).exists() {
            return "CUDA".to_string();
        }
    }

    // Check for Vulkan DLL
    if dir.join("ggml-vulkan.dll").exists() {
        return "Vulkan".to_string();
    }

    // Check path string as last resort
    let ps = path.to_string_lossy().to_lowercase();
    if ps.contains("cuda") {
        "CUDA".to_string()
    } else if ps.contains("vulkan") {
        "Vulkan".to_string()
    } else {
        "GPU".to_string()
    }
}

fn detect_api_port_owner(port: u16) -> Option<ApiPortOwnerInfo> {
    #[cfg(windows)]
    {
        return detect_api_port_owner_windows(port);
    }

    #[allow(unreachable_code)]
    None
}

#[cfg(windows)]
fn detect_api_port_owner_windows(port: u16) -> Option<ApiPortOwnerInfo> {
    let current_pid = std::process::id();
    let output = command_no_window("netstat")
        .args(["-ano", "-p", "tcp"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let port_suffix = format!(":{port}");
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let columns: Vec<&str> = line.split_whitespace().collect();
        if columns.len() < 5 {
            continue;
        }

        let proto = columns[0];
        let local_addr = columns[1];
        let state = columns[3];
        let Ok(pid) = columns[4].parse::<u32>() else {
            continue;
        };

        if !proto.eq_ignore_ascii_case("TCP")
            || !local_addr.ends_with(&port_suffix)
            || state != "LISTENING"
        {
            continue;
        }

        let name = process_name_for_pid(pid);
        let lower = name.as_deref().unwrap_or_default().to_lowercase();
        let (kind, killable) = if pid == current_pid {
            ("self".to_string(), false)
        } else if lower.contains("llama-server") {
            ("llama-server".to_string(), true)
        } else if lower.contains("inference-bridge") {
            ("inference-bridge".to_string(), true)
        } else if lower.is_empty() {
            return None;
        } else {
            ("other".to_string(), false)
        };

        return Some(ApiPortOwnerInfo {
            pid,
            name,
            kind,
            killable,
        });
    }

    None
}

#[cfg(windows)]
fn process_name_for_pid(pid: u32) -> Option<String> {
    let filter = format!("PID eq {pid}");
    let output = command_no_window("tasklist")
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    if line.is_empty() || line.starts_with("INFO:") {
        return None;
    }

    let parts: Vec<&str> = line.split(',').collect();
    let name = parts.first()?.trim_matches('"').trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Info about a running llama-server process on this machine.
#[derive(serde::Serialize)]
pub struct ExternalProcess {
    pub pid: u32,
    pub name: String,
    pub command_line: String,
    pub memory_mb: f64,
}

#[tauri::command]
pub async fn list_llama_processes() -> Result<Vec<ExternalProcess>, String> {
    // Use WMIC to list all llama-server processes
    let output = command_no_window("wmic")
        .args([
            "process",
            "where",
            "name like '%llama%server%'",
            "get",
            "ProcessId,Name,CommandLine,WorkingSetSize",
            "/FORMAT:CSV",
        ])
        .output()
        .map_err(|e| format!("Failed to query processes: {e}"))?;

    if !output.status.success() {
        // Fallback: use tasklist
        return list_llama_processes_tasklist();
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut processes = Vec::new();
    for line in text.lines().skip(1) {
        // CSV: Node,CommandLine,Name,ProcessId,WorkingSetSize
        // CommandLine can itself contain commas, so split from the RIGHT to safely
        // extract the fixed-position trailing fields.
        let parts_rev: Vec<&str> = line.rsplitn(5, ',').collect();
        // rsplitn gives: [WorkingSetSize, ProcessId, Name, CommandLine..., Node] (rev order)
        if parts_rev.len() >= 5 {
            let mem_bytes: f64 = parts_rev[0].trim().parse().unwrap_or(0.0);
            let pid: u32 = parts_rev[1].trim().parse().unwrap_or(0);
            let name = parts_rev[2].trim().to_string();
            let cmd = parts_rev[3].trim().to_string(); // may include commas from original
            if pid > 0
                && (name.to_lowercase().contains("llama") || cmd.to_lowercase().contains("llama"))
            {
                processes.push(ExternalProcess {
                    pid,
                    name,
                    command_line: cmd,
                    memory_mb: mem_bytes / (1024.0 * 1024.0),
                });
            }
        }
    }
    Ok(processes)
}

fn list_llama_processes_tasklist() -> Result<Vec<ExternalProcess>, String> {
    let output = command_no_window("tasklist")
        .args(["/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"])
        .output()
        .map_err(|e| format!("Failed to query processes: {e}"))?;

    let text = String::from_utf8_lossy(&output.stdout);
    let mut processes = Vec::new();
    for line in text.lines() {
        // CSV: "name","pid","session","session#","mem"
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 5 {
            let name = parts[0].trim_matches('"').to_string();
            let pid: u32 = parts[1].trim_matches('"').trim().parse().unwrap_or(0);
            let mem_str = parts[4]
                .trim_matches('"')
                .trim()
                .replace(" K", "")
                .replace(",", "")
                .replace(".", "");
            let mem_kb: f64 = mem_str.parse().unwrap_or(0.0);
            if pid > 0 {
                processes.push(ExternalProcess {
                    pid,
                    name,
                    command_line: String::new(),
                    memory_mb: mem_kb / 1024.0,
                });
            }
        }
    }
    Ok(processes)
}

#[tauri::command]
pub async fn kill_process(pid: u32) -> Result<String, String> {
    let output = command_no_window("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .output()
        .map_err(|e| format!("Failed to kill process: {e}"))?;

    if output.status.success() {
        tracing::info!(pid = pid, "Killed process");
        Ok(format!("Killed process {pid}"))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Failed to kill PID {pid}: {stderr}"))
    }
}

#[tauri::command]
pub async fn kill_all_llama_processes() -> Result<String, String> {
    let procs = list_llama_processes().await?;
    if procs.is_empty() {
        return Ok("No llama-server processes found".to_string());
    }

    let mut killed = 0u32;
    let mut errors = Vec::new();
    for proc in &procs {
        match kill_process(proc.pid).await {
            Ok(_) => killed += 1,
            Err(e) => errors.push(e),
        }
    }

    if errors.is_empty() {
        Ok(format!("Killed {killed} process(es)"))
    } else {
        Ok(format!("Killed {killed}, errors: {}", errors.join("; ")))
    }
}

/// GPU VRAM usage stats queried from nvidia-smi.
#[derive(serde::Serialize)]
pub struct GpuStats {
    pub name: String,
    pub used_mb: u64,
    /// Dedicated VRAM (the fast on-board memory, from nvidia-smi memory.total).
    pub dedicated_mb: u64,
    pub free_mb: u64,
    /// Total system RAM — shown as the "overflow/spill" zone beyond dedicated VRAM.
    pub system_ram_mb: u64,
}

fn get_system_ram_mb() -> u64 {
    let output = command_no_window("wmic")
        .args(["OS", "get", "TotalVisibleMemorySize", "/VALUE"])
        .output();
    if let Ok(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if let Some(val) = line.strip_prefix("TotalVisibleMemorySize=") {
                if let Ok(kb) = val.trim().parse::<u64>() {
                    return kb / 1024;
                }
            }
        }
    }
    0
}

#[tauri::command]
pub async fn get_gpu_stats() -> Result<GpuStats, String> {
    let output = command_no_window("nvidia-smi")
        .args([
            "--query-gpu=name,memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| format!("nvidia-smi not available: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("nvidia-smi error: {stderr}"));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next().ok_or("No output from nvidia-smi")?;
    let parts: Vec<&str> = line.splitn(4, ',').collect();
    if parts.len() < 4 {
        return Err(format!("Unexpected nvidia-smi output: {line}"));
    }

    Ok(GpuStats {
        name: parts[0].trim().to_string(),
        used_mb: parts[1].trim().parse().unwrap_or(0),
        dedicated_mb: parts[2].trim().parse().unwrap_or(0),
        free_mb: parts[3].trim().parse().unwrap_or(0),
        system_ram_mb: get_system_ram_mb(),
    })
}

/// Extract quantization level from a GGUF filename (e.g. "Q4_K_M", "Q5_0", "IQ4_XS").
fn extract_quant(filename: &str) -> Option<String> {
    let upper = filename.to_uppercase();
    // Match common GGUF quant patterns: Q4_K_M, Q5_0, IQ4_XS, F16, etc.
    let re = regex::Regex::new(
        r"[_.-]((?:I?Q\d+_[A-Z0-9]+(?:_[A-Z]+)?)|F(?:16|32)|BF16)(?:[_.-]|\.GGUF$)",
    )
    .ok()?;
    re.captures(&upper).map(|c| c[1].to_string())
}

#[derive(serde::Serialize)]
pub struct ModelInfo {
    pub filename: String,
    pub path: String,
    pub size_gb: f64,
    pub family: String,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub supports_vision: bool,
    pub context_window: Option<u32>,
    pub max_context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub default_temperature: Option<f32>,
    pub default_top_p: Option<f32>,
    pub default_top_k: Option<i32>,
    pub default_min_p: Option<f32>,
    pub default_presence_penalty: Option<f32>,
    pub quant: Option<String>,
    pub tool_call_format: String,
    pub think_tag_style: String,
    pub hf_repo: Option<String>,
    pub hf_file: Option<String>,
    pub template_mode: Option<String>,
    pub template_source: Option<String>,
    pub vision_runtime_ready: bool,
    pub vision_status: String,
    pub provider_type: String,
    pub provider_name: String,
    pub provider_base_url: Option<String>,
    pub provider_managed: bool,
    // GGUF architecture metadata — populated for locally-scanned models
    pub n_layers: Option<u32>,
    pub n_kv_heads: Option<u32>,
    pub head_dim: Option<u32>,
    pub gguf_architecture: Option<String>,
}

#[derive(serde::Serialize)]
pub struct ProcessStatusInfo {
    pub state: String,
    pub model: Option<String>,
    pub previous_model: Option<String>,
    pub model_load_state: String,
    pub model_load_progress: Option<LoadProgress>,
    pub active_generation: Option<GenerationRequest>,
    pub crash_count: u32,
    pub server_version: Option<String>,
    pub server_path: Option<String>,
    pub backend: Option<String>,
    pub api_state: String,
    pub api_error: Option<String>,
    pub api_url: String,
    pub api_reachable: bool,
    pub api_port_owner: Option<ApiPortOwnerInfo>,
    pub startup_duration_ms: Option<u64>,
    pub parallel_slots: Option<u32>,
    pub slot_count: Option<u32>,
    pub active_requests: usize,
    pub queued_requests: usize,
    pub scheduler_limit: Option<u32>,
    pub last_launch_preview: Option<LaunchPreview>,
    pub last_generation_metrics: Option<RuntimePerformanceMetrics>,
    pub live_stream: Option<crate::state::LiveStreamSnapshot>,
}

#[derive(Clone, serde::Serialize)]
pub struct ApiPortOwnerInfo {
    pub pid: u32,
    pub name: Option<String>,
    pub kind: String,
    pub killable: bool,
}

#[derive(serde::Serialize)]
pub struct ServerInfo {
    pub found: bool,
    pub path: Option<String>,
    pub version: Option<String>,
}

#[tauri::command]
pub async fn get_effective_profile(
    state: tauri::State<'_, SharedState>,
    model_name: Option<String>,
) -> Result<EffectiveProfileInfo, String> {
    let s = state.read().await;
    effective_profile_info_from_state(&s, model_name.as_deref())
}

#[tauri::command]
pub async fn set_model_vision_override(
    state: tauri::State<'_, SharedState>,
    model_name: String,
    supports_vision: bool,
) -> Result<(), String> {
    crate::models::overrides::set_model_supports_vision_override(&model_name, supports_vision)?;

    // Re-scan so the in-memory registry reflects the new override immediately.
    let mut s = state.write().await;
    let models = crate::models::scanner::scan_all(&s.config.models.scan_dirs);
    s.model_registry.update(models);
    Ok(())
}

#[tauri::command]
pub async fn reload_last_known_good(
    state: tauri::State<'_, SharedState>,
) -> Result<String, String> {
    let preview = {
        let s = state.read().await;
        s.last_known_good_config.clone()
    }
    .ok_or_else(|| "No last known good launch configuration has been recorded yet".to_string())?;

    let model_name = std::path::Path::new(&preview.model_path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .ok_or_else(|| "Last known good config does not contain a valid model path".to_string())?;

    if !std::path::Path::new(&preview.model_path).exists() {
        return Err(format!(
            "The last known good model path no longer exists: {}",
            preview.model_path
        ));
    }

    backend_load_model(state.inner().clone(), model_name, None).await
}

#[tauri::command]
pub async fn check_llama_server(
    state: tauri::State<'_, SharedState>,
) -> Result<ServerInfo, String> {
    let s = state.read().await;
    let path = s.process.find_server_binary();
    let version = download::current_version();
    Ok(ServerInfo {
        found: path.is_some(),
        path: path.map(|p| p.to_string_lossy().to_string()),
        version,
    })
}

#[tauri::command]
pub async fn update_llama_server(app: tauri::AppHandle) -> Result<String, String> {
    match download::check_for_update().await {
        Ok(Some((tag, url, size))) => {
            match download::download_llama_server(&app, &url, &tag, size).await {
                Ok(path) => {
                    emit_load_progress_payload(
                        &Some(app.clone()),
                        &crate::state::LoadProgress {
                            stage: "ready".to_string(),
                            message: format!("Updated llama-server to {tag}"),
                            progress: 1.0,
                            done: true,
                            error: None,
                        },
                    );
                    Ok(format!("Updated to {} at {}", tag, path.display()))
                }
                Err(e) => Err(format!("Download failed: {e}")),
            }
        }
        Ok(None) => Ok("Already up-to-date".to_string()),
        Err(e) => Err(format!("Update check failed: {e}")),
    }
}

/// Swap to a different model. If no model_name is given, swaps back to the
/// previously loaded model. This is the same as load_model but with
/// swap-back semantics.
#[tauri::command]
pub async fn swap_model(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
    model_name: Option<String>,
    options: Option<RuntimeLoadRequest>,
) -> Result<String, String> {
    // Determine target: explicit name or previous model
    let target = match model_name {
        Some(name) => name,
        None => {
            let s = state.read().await;
            s.previous_model
                .clone()
                .ok_or_else(|| "No previous model to swap back to".to_string())?
        }
    };

    tracing::info!(target = %target, "Hot-swap requested");

    let _ = app;
    let options = options.unwrap_or_default();
    backend_load_model_with_overrides(
        state.inner().clone(),
        target,
        options.normalized_context_size(),
        options.into_overrides(),
    )
    .await
}

import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  ArrowUp,
  Box,
  ChevronDown,
  Image as ImageIcon,
  Paperclip,
  Plus,
  RotateCcw,
  SlidersHorizontal,
  Square,
  X,
} from "lucide-react";
import type {
  ImageGenerationCapabilityStatus,
  ImageGenerationProgress,
  ImageSizePreset,
  MessageInfo,
  ModelInfo,
  ProcessStatusInfo,
} from "../../lib/types";
import {
  cancelImageGeneration,
  generateImage,
  getImageGenerationStatus,
  type SamplingParams,
} from "../../lib/tauri";
import { readSamplingSettings, recommendedLoadPresets } from "../../lib/modelLoadProfiles";
import { latestAssistantOutputTokens } from "../../lib/chatPresentation";
import { composerPrimaryAction, isNearScrollBottom } from "../../lib/conversationUi";
import {
  PRESETS,
  PRESET_ORDER,
  modelSupportsThinking,
} from "../../lib/presets";
import { Button, IconButton } from "../ui/Controls";
import { ModelArtwork, modelDisplayName } from "../Model/modelPresentation";
import { MessageBubble } from "./MessageBubble";
import { StreamingText } from "./StreamingText";
import { CanvasPanel, type CanvasVersion } from "./CanvasPanel";

interface Props {
  messages: MessageInfo[];
  isStreaming: boolean;
  streamingText: string;
  streamingReasoning: string;
  tokensPerSecond: number | null;
  processStatus?: ProcessStatusInfo | null;
  error: string | null;
  hasModel: boolean;
  hasSession: boolean;
  sessionId?: string | null;
  loadedModel?: string | null;
  loadedModelInfo?: ModelInfo | null;
  modelPickerOpen?: boolean;
  loadedModelVisionConfigured?: boolean;
  loadedModelSupportsVision?: boolean;
  loadedModelVisionStatusText?: string | null;
  onSend: (
    content: string,
    sampling?: SamplingParams,
    imageBase64?: string | null,
    showThinking?: boolean | null
  ) => void;
  onStop: () => void;
  canCreateSession: boolean;
  creatingSession: boolean;
  onCreateSession: () => void;
  onOpenModelPicker: (trigger: HTMLElement | null) => void;
  onOpenImageSettings: () => void;
}

const suggestionPrompts = [
  "Explain this error and suggest a fix",
  "Draft a robust tool-call schema",
  "Help me compare two implementation options",
];

export function ChatPanel({
  messages,
  isStreaming,
  streamingText,
  streamingReasoning,
  tokensPerSecond,
  processStatus = null,
  error,
  hasModel,
  hasSession,
  sessionId = null,
  loadedModel,
  loadedModelInfo = null,
  modelPickerOpen = false,
  loadedModelVisionConfigured = false,
  loadedModelSupportsVision = false,
  loadedModelVisionStatusText = null,
  onSend,
  onStop,
  canCreateSession,
  creatingSession,
  onCreateSession,
  onOpenModelPicker,
  onOpenImageSettings,
}: Props) {
  const [input, setInput] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [controlsOpen, setControlsOpen] = useState(false);
  const [sampling, setSampling] = useState<SamplingParams>({});
  const [showThinking, setShowThinking] = useState(false);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [composerError, setComposerError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [canvasVersions, setCanvasVersions] = useState<CanvasVersion[]>([]);
  const [canvasIndex, setCanvasIndex] = useState(0);
  const [canvasOpen, setCanvasOpen] = useState(false);
  const [followingOutput, setFollowingOutput] = useState(true);
  const [imageProgress, setImageProgress] = useState<ImageGenerationProgress | null>(null);
  const [imageCapability, setImageCapability] = useState<ImageGenerationCapabilityStatus | null>(null);
  const [imageControlsOpen, setImageControlsOpen] = useState(false);
  const [imageSubmitting, setImageSubmitting] = useState(false);
  const [imageProgressReceivedAt, setImageProgressReceivedAt] = useState(Date.now());
  const [, setImageClock] = useState(Date.now());
  const bottomRef = useRef<HTMLDivElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const followingOutputRef = useRef(true);
  const lastScrollTopRef = useRef(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const controlsRef = useRef<HTMLDivElement>(null);
  const imageControlsRef = useRef<HTMLDivElement>(null);

  const canThink = loadedModelInfo?.supports_reasoning ?? modelSupportsThinking(loadedModel);
  const tessQwenPresets = loadedModelInfo ? recommendedLoadPresets(loadedModelInfo) : [];
  const generationPresets = tessQwenPresets.length > 0
    ? tessQwenPresets.map((preset) => ({
        key: preset.id,
        label: preset.id === "general-thinking" ? "General sampler" : preset.name,
        description: `${preset.description} Sampler only; the loaded --reasoning mode is unchanged.`,
        sampling: {
          temperature: preset.sampling.temperature,
          top_p: preset.sampling.topP,
          top_k: preset.sampling.topK,
          min_p: preset.sampling.minP,
          presence_penalty: preset.sampling.presencePenalty,
          repeat_penalty: preset.sampling.repeatPenalty,
        } satisfies SamplingParams,
        suggestThinking: null,
      }))
    : PRESET_ORDER.map((key) => ({ key, ...PRESETS[key] }));
  const activePresetConfig = generationPresets.find((preset) => preset.key === activePreset) ?? null;
  const lastMetrics = processStatus?.last_generation_metrics ?? null;
  const activeGeneration = processStatus?.active_generation ?? null;
  const liveGeneratedApprox = Math.max(
    0,
    Math.round((streamingText.length + streamingReasoning.length) / 4)
  );
  const latestStoredOutputTokens = latestAssistantOutputTokens(messages);
  const generatedTokens = isStreaming
    ? liveGeneratedApprox
    : latestStoredOutputTokens ?? lastMetrics?.completion_tokens ?? null;
  const lastDecodeRate = lastMetrics?.decode_tokens_per_second;
  const displayTokSec = isStreaming
    ? tokensPerSecond
    : tokensPerSecond ?? (lastDecodeRate != null && lastDecodeRate > 0 ? lastDecodeRate : null);
  const hasCustomSampling = Object.values(sampling).some(
    (value) => value !== undefined
  );
  const imageJobActive = !!imageProgress && !imageProgress.done;
  const primaryAction = composerPrimaryAction(hasModel, !!imageCapability?.ready);
  const canSubmit =
    hasSession &&
    !isStreaming &&
    !imageJobActive &&
    !imageSubmitting &&
    (
      primaryAction === "send_message"
        ? !!input.trim() || !!image
        : primaryAction === "generate_image"
          ? !!input.trim() && !image
          : false
    );
  const sendDisabledReason = isStreaming
    ? "Generation is already running"
    : imageJobActive
      ? "Image generation is using the GPU"
      : imageSubmitting
        ? "Image generation is starting"
        : !hasSession
          ? "Select a conversation before sending"
          : primaryAction === "unavailable"
            ? "Set up image generation in Settings"
            : primaryAction === "generate_image" && image
              ? "Remove the attachment before generating an image"
              : !input.trim() && !image
                ? primaryAction === "generate_image"
                  ? "Describe the image you want to generate"
                  : "Enter a message or attach an image"
                : undefined;

  const openCanvas = useCallback((html: string) => {
    setCanvasVersions((current) => {
      const existing = current.findIndex((version) => version.html === html);
      if (existing >= 0) {
        setCanvasIndex(existing);
        return current;
      }
      const next = [...current, { html, label: `Version ${current.length + 1}` }].slice(-12);
      setCanvasIndex(next.length - 1);
      return next;
    });
    setCanvasOpen(true);
  }, []);

  const updateFollowingOutput = useCallback((next: boolean) => {
    followingOutputRef.current = next;
    setFollowingOutput(next);
  }, []);

  const scrollToLatest = useCallback((resumeFollowing = true) => {
    if (resumeFollowing) updateFollowingOutput(true);
    const node = chatScrollRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
    lastScrollTopRef.current = node.scrollTop;
  }, [updateFollowingOutput]);

  useEffect(() => {
    if (!followingOutputRef.current) return undefined;
    const frame = window.requestAnimationFrame(() => scrollToLatest(false));
    return () => window.cancelAnimationFrame(frame);
  }, [messages.length, scrollToLatest, streamingReasoning.length, streamingText.length]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(200, Math.max(28, textarea.scrollHeight))}px`;
  }, [input]);

  useEffect(() => {
    setInput("");
    setImage(null);
    setImagePreview(null);
    setComposerError(null);
    setControlsOpen(false);
    updateFollowingOutput(true);
    lastScrollTopRef.current = 0;
    const frame = window.requestAnimationFrame(() => scrollToLatest(false));
    return () => window.cancelAnimationFrame(frame);
  }, [scrollToLatest, sessionId, updateFollowingOutput]);

  useEffect(() => {
    const focusComposer = () => textareaRef.current?.focus();
    window.addEventListener("ib-focus-composer", focusComposer);
    return () => window.removeEventListener("ib-focus-composer", focusComposer);
  }, []);

  useEffect(() => {
    if (!controlsOpen) return undefined;
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (!controlsRef.current?.contains(event.target as Node)) {
        setControlsOpen(false);
      }
    };
    window.addEventListener("mousedown", closeOnOutsideClick);
    return () => window.removeEventListener("mousedown", closeOnOutsideClick);
  }, [controlsOpen]);

  useEffect(() => {
    let mounted = true;
    let stopListening: (() => void) | undefined;
    const receiveProgress = (progress: ImageGenerationProgress | null) => {
      if (!mounted || !progress) return;
      setImageProgress(progress);
      setImageProgressReceivedAt(Date.now());
    };

    const refreshCapability = () => {
      void getImageGenerationStatus()
        .then((status) => {
          if (!mounted) return;
          setImageCapability(status);
          receiveProgress(status.active_job);
        })
        .catch(() => {
          // The browser-only Vite preview has no native image runtime.
        });
    };
    refreshCapability();
    window.addEventListener("ib-image-settings-updated", refreshCapability);
    void listen<ImageGenerationProgress>("image-generation-progress", (event) => {
      receiveProgress(event.payload);
    }).then((unlisten) => {
      if (mounted) stopListening = unlisten;
      else unlisten();
    });

    return () => {
      mounted = false;
      stopListening?.();
      window.removeEventListener("ib-image-settings-updated", refreshCapability);
    };
  }, []);

  useEffect(() => {
    if (!imageControlsOpen) return undefined;
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (!imageControlsRef.current?.contains(event.target as Node)) {
        setImageControlsOpen(false);
      }
    };
    window.addEventListener("mousedown", closeOnOutsideClick);
    return () => window.removeEventListener("mousedown", closeOnOutsideClick);
  }, [imageControlsOpen]);

  useEffect(() => {
    if (!imageProgress || imageProgress.done) return undefined;
    const timer = window.setInterval(() => setImageClock(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [imageProgress?.done, imageProgress?.job_id]);

  useEffect(() => {
    const preview = processStatus?.last_launch_preview;
    if (!loadedModel || !preview) return;
    const fromArgs = readSamplingSettings(preview.args ?? []);
    const defaults = preview.sampling_defaults;
    const nextSampling: SamplingParams = {
      temperature: defaults?.temperature ?? fromArgs.temperature ?? undefined,
      top_p: defaults?.top_p ?? fromArgs.topP ?? undefined,
      top_k: defaults?.top_k ?? fromArgs.topK ?? undefined,
      min_p: defaults?.min_p ?? fromArgs.minP ?? undefined,
      presence_penalty: defaults?.presence_penalty ?? fromArgs.presencePenalty ?? undefined,
      repeat_penalty: defaults?.repeat_penalty ?? fromArgs.repeatPenalty ?? undefined,
    };
    setSampling(nextSampling);
    setActivePreset(null);
    if (canThink && preview.reasoning_mode && preview.reasoning_mode !== "auto") {
      setShowThinking(preview.reasoning_mode === "on");
    }
  }, [canThink, loadedModel, processStatus?.last_launch_preview?.model_path]);

  const applyPreset = (key: string) => {
    const preset = generationPresets.find((candidate) => candidate.key === key);
    if (!preset) return;
    setActivePreset(key);
    setSampling(preset.sampling);
  };

  const resetControls = () => {
    setActivePreset(null);
    setSampling({});
    setShowThinking(false);
  };

  const attachImage = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setComposerError("Only image attachments are supported in chat.");
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      setComposerError("Images must be 16 MB or smaller.");
      return;
    }

    setImage(file);
    setComposerError(null);
    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result;
      if (typeof result === "string") setImagePreview(result);
    };
    reader.onerror = () => {
      setImage(null);
      setImagePreview(null);
      setComposerError("InferenceBridge could not read that image.");
    };
    reader.readAsDataURL(file);
  };

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) attachImage(file);
    event.target.value = "";
  };

  const removeImage = () => {
    setImage(null);
    setImagePreview(null);
    setComposerError(null);
  };

  const handlePaste = (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const imageItem = Array.from(event.clipboardData.items).find((item) =>
      item.type.startsWith("image/")
    );
    if (!imageItem) return;
    const file = imageItem.getAsFile();
    if (!file) return;
    event.preventDefault();
    attachImage(file);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    const file = Array.from(event.dataTransfer.files).find((item) =>
      item.type.startsWith("image/")
    );
    if (file) attachImage(file);
  };

  const openImageControls = async () => {
    try {
      const status = await getImageGenerationStatus();
      setImageCapability(status);
      if (!status.ready) {
        setComposerError("Set up and enable Qwen image generation in Settings first.");
        onOpenImageSettings();
        return;
      }
      setComposerError(null);
      setImageControlsOpen(true);
    } catch (imageError) {
      setComposerError(String(imageError));
    }
  };

  const handleGenerateImage = (options: {
    size: ImageSizePreset;
    steps: number;
    cfgScale: number;
    sampler: string;
    seed: number | null;
    negativePrompt: string;
  }) => {
    const prompt = input.trim();
    if (!prompt || !sessionId || imageSubmitting || imageJobActive) return;
    setImageSubmitting(true);
    setImageControlsOpen(false);
    setComposerError(null);
    setInput("");
    void generateImage({
      prompt,
      session_id: sessionId,
      profile_id: "quality",
      width: options.size.width,
      height: options.size.height,
      steps: options.steps,
      cfg_scale: options.cfgScale,
      sampling_method: options.sampler,
      seed: options.seed,
      negative_prompt: options.negativePrompt || null,
    }).then(
      (result) => {
        if (result.error) setComposerError(result.error);
      },
      (imageError) => {
        setInput(prompt);
        setComposerError(String(imageError));
      },
    ).finally(() => setImageSubmitting(false));
  };

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!hasSession || isStreaming || imageJobActive || imageSubmitting) return;

    if (primaryAction === "generate_image") {
      const defaultSize =
        imageCapability?.size_presets.find((preset) => preset.id === "recommended_square") ??
        imageCapability?.size_presets[0];
      if (!trimmed || image || !defaultSize) return;
      handleGenerateImage({
        size: defaultSize,
        steps: 50,
        cfgScale: 2.5,
        sampler: "euler",
        seed: null,
        negativePrompt: "",
      });
      return;
    }

    if (primaryAction !== "send_message" || (!trimmed && !image)) return;
    if (image && !loadedModelSupportsVision) {
      setComposerError(
        loadedModelVisionStatusText ??
          "The current model is not vision-ready. Load a vision model with its matching mmproj sidecar first."
      );
      return;
    }

    const params = hasCustomSampling ? sampling : undefined;
    setComposerError(null);
    scrollToLatest();
    onSend(trimmed, params, imagePreview, showThinking);
    setInput("");
    setImage(null);
    setImagePreview(null);
  };

  const chooseSuggestion = (prompt: string) => {
    setInput(prompt);
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      <div className="flex h-12 shrink-0 items-center gap-3 border-b border-[var(--border)] px-4">
        <button
          type="button"
          data-context-copy={loadedModelInfo?.filename ?? loadedModel ?? undefined}
          data-context-label="model name"
          onClick={(event) => onOpenModelPicker(event.currentTarget)}
          aria-label={loadedModelInfo ? `Change model. Current model ${modelDisplayName(loadedModelInfo)}` : loadedModel ? `Change model. Current model ${loadedModel}` : "Choose a model"}
          aria-haspopup="dialog"
          aria-expanded={modelPickerOpen}
          aria-controls="rich-model-picker"
          className="flex min-w-0 items-center gap-2 rounded-lg px-2 py-1.5 text-left hover:bg-white/5"
          title="Choose a model"
        >
          {loadedModelInfo ? (
            <ModelArtwork model={loadedModelInfo} size="xs" />
          ) : (
            <Box size={15} className={loadedModel ? "text-emerald-400" : "text-[var(--text-3)]"} />
          )}
          <span className="max-w-[460px] truncate text-sm font-medium text-[var(--text-0)]">
            {loadedModelInfo ? modelDisplayName(loadedModelInfo) : loadedModel ?? "Choose a model"}
          </span>
          <ChevronDown size={14} className={`text-[var(--text-3)] transition ${modelPickerOpen ? "rotate-180" : ""}`} />
        </button>

        <div className="ml-auto flex items-center gap-3 text-[11px] tabular-nums text-[var(--text-2)]">
          {isStreaming && activeGeneration && (
            <span className="hidden items-center gap-1.5 sm:flex">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
              {activeGeneration.status}
            </span>
          )}
          {generatedTokens != null && <span>{generatedTokens.toLocaleString()} tokens</span>}
          {displayTokSec != null && (
            <span className="text-emerald-300">
              {displayTokSec.toFixed(displayTokSec >= 100 ? 0 : 1)} tok/s
            </span>
          )}
          {isStreaming && displayTokSec == null && (
            <span className="text-[var(--text-3)]">measuring speed…</span>
          )}
        </div>
      </div>

      <div className="relative min-h-0 flex-1">
        <div
          ref={chatScrollRef}
          className="h-full overflow-y-auto"
          aria-live="polite"
          onScroll={(event) => {
            const node = event.currentTarget;
            const scrollingUp = node.scrollTop < lastScrollTopRef.current - 1;
            const nearBottom = isNearScrollBottom(node.scrollHeight, node.scrollTop, node.clientHeight);
            lastScrollTopRef.current = node.scrollTop;

            if (scrollingUp) {
              updateFollowingOutput(false);
            } else if (nearBottom) {
              updateFollowingOutput(true);
            } else if (followingOutputRef.current) {
              updateFollowingOutput(false);
            }
          }}
        >
          {!hasModel && !(hasSession && imageCapability?.ready) ? (
          <ChatEmptyState
            icon={<Box size={22} />}
            title="Load a model to begin"
            action={<Button onClick={(event) => onOpenModelPicker(event.currentTarget)}>Choose a model</Button>}
          />
        ) : !hasSession ? (
          <ChatEmptyState
            title="Select a chat to continue"
            description="Or start a fresh conversation with the selected model."
            action={(
              <Button
                variant="primary"
                icon={<Plus size={16} />}
                disabled={!canCreateSession || creatingSession}
                onClick={onCreateSession}
              >
                {creatingSession ? "Creating chat..." : "New chat"}
              </Button>
            )}
          />
        ) : messages.length === 0 && !isStreaming && !imageProgress ? (
          <div className="mx-auto flex h-full w-full max-w-[760px] flex-col items-center justify-center px-6 pb-20 text-center">
            <div className="ib-brand-mark mb-5 h-10 w-10 text-xs">IB</div>
            <h1 className="text-2xl font-semibold text-[var(--text-0)]">What are we working on?</h1>
            <div className="mt-7 grid w-full grid-cols-1 gap-2 sm:grid-cols-3">
              {suggestionPrompts.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => chooseSuggestion(prompt)}
                  className="min-h-[72px] rounded-xl border border-[var(--border)] bg-[var(--surface-1)] px-4 py-3 text-left text-sm leading-5 text-[var(--text-1)] transition hover:bg-[var(--surface-2)] hover:text-[var(--text-0)]"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto w-full max-w-[var(--content-max)] px-4 pb-8 pt-5 sm:px-8">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} onOpenHtml={openCanvas} />
            ))}
            {imageProgress && (
              <ImageGenerationProgressCard
                progress={imageProgress}
                receivedAt={imageProgressReceivedAt}
                onCancel={() => {
                  void cancelImageGeneration();
                }}
              />
            )}
            {isStreaming && (
              <StreamingText text={streamingText} reasoning={streamingReasoning} onOpenHtml={openCanvas} />
            )}
            <div ref={bottomRef} className="h-2" />
          </div>
          )}
        </div>
        {hasSession && !followingOutput && (messages.length > 0 || isStreaming || !!imageProgress) && (
          <button
            type="button"
            className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-white/10 bg-[var(--surface-3)] px-3 py-1.5 text-xs font-medium text-[var(--text-1)] shadow-lg transition hover:bg-[var(--surface-hover)] hover:text-[var(--text-0)]"
            onClick={() => scrollToLatest()}
            aria-label="Jump to latest message and resume following output"
          >
            <ChevronDown size={14} />
            Jump to latest
          </button>
        )}
      </div>

      {hasSession && (hasModel || imageCapability?.ready) && (
        <div className="shrink-0 px-3 pb-3 pt-2 sm:px-6 sm:pb-4">
          <div className="mx-auto w-full max-w-[var(--content-max)]">
            {(error || composerError) && (
              <div className="mb-2 rounded-lg border border-rose-400/20 bg-rose-950/25 px-3 py-2 text-xs leading-5 text-rose-200">
                {composerError ?? error}
              </div>
            )}

            <div
              className={`relative rounded-[22px] border bg-[var(--surface-3)] shadow-[0_8px_32px_rgba(0,0,0,0.18)] transition-colors ${
                dragActive
                  ? "border-white/40 bg-[var(--surface-hover)]"
                  : "border-white/10"
              }`}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget as Node)) {
                  setDragActive(false);
                }
              }}
              onDrop={handleDrop}
            >
              {dragActive && (
                <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-[22px] bg-[#303030]/95 text-sm font-medium text-white">
                  <ImageIcon size={18} className="mr-2" />
                  Drop image
                </div>
              )}

              {imagePreview && (
                <div className="flex items-center gap-3 px-3 pt-3">
                  <div className="relative h-16 w-16 shrink-0 overflow-hidden rounded-xl border border-white/10 bg-black/20">
                    <img src={imagePreview} alt="Chat attachment" className="h-full w-full object-cover" />
                    <button
                      type="button"
                      aria-label="Remove image"
                      title="Remove image"
                      className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-black/70 text-white hover:bg-black"
                      onClick={removeImage}
                    >
                      <X size={12} />
                    </button>
                  </div>
                  <div className="min-w-0">
                    <div className="truncate text-xs font-medium text-[var(--text-0)]">
                      {image?.name || "Pasted image"}
                    </div>
                    <div className={`mt-1 text-[11px] ${loadedModelSupportsVision ? "text-emerald-300" : "text-amber-300"}`}>
                      {loadedModelSupportsVision
                        ? "Vision ready"
                        : loadedModelVisionConfigured
                          ? "Projector not attached"
                          : "Vision model required"}
                    </div>
                    {image && <div className="mt-1 text-[10px] text-[var(--text-3)]">{(image.size / 1024).toFixed(image.size >= 1024 * 1024 ? 0 : 1)} KB · {image.type || "image"}</div>}
                  </div>
                </div>
              )}

              <textarea
                ref={textareaRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    handleSubmit();
                  }
                  if (event.key === "Escape" && isStreaming) {
                    event.preventDefault();
                    onStop();
                  }
                }}
                onPaste={handlePaste}
                placeholder={hasModel ? "Message InferenceBridge" : "Describe the image you want to generate"}
                rows={1}
                className="ib-chat-input block max-h-[200px] min-h-[46px] w-full resize-none overflow-y-auto bg-transparent px-4 pb-2 pt-3.5 text-[15px] leading-6 text-[var(--text-0)] outline-none placeholder:text-[var(--text-3)]"
              />

              <div className="flex h-12 items-center justify-between gap-2 px-2 pb-1.5">
                <div className="flex items-center gap-0.5">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleImageChange}
                    disabled={isStreaming || !hasModel}
                  />
                  <IconButton
                    label="Attach image"
                    size="md"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isStreaming || !hasModel}
                  >
                    <Paperclip size={18} />
                  </IconButton>

                  <div className="relative" ref={imageControlsRef}>
                    <IconButton
                      label={imageCapability?.ready ? "Generate image" : "Set up image generation"}
                      size="md"
                      selected={imageControlsOpen}
                      onClick={() => { void openImageControls(); }}
                      disabled={isStreaming || imageJobActive || imageSubmitting || !input.trim()}
                    >
                      <ImageIcon size={18} />
                    </IconButton>
                    {imageControlsOpen && imageCapability?.ready && (
                      <ImageGenerationControls
                        presets={imageCapability.size_presets}
                        busy={imageSubmitting || imageJobActive}
                        requiresManualUnload={
                          hasModel && !imageCapability.automatic_model_swap_enabled
                        }
                        onGenerate={handleGenerateImage}
                        onClose={() => setImageControlsOpen(false)}
                      />
                    )}
                  </div>

                  <div className="relative" ref={controlsRef}>
                    <IconButton
                      label="Generation controls"
                      size="md"
                      selected={controlsOpen || hasCustomSampling || showThinking}
                      onClick={() => setControlsOpen((value) => !value)}
                      disabled={!hasModel}
                    >
                      <SlidersHorizontal size={17} />
                    </IconButton>

                    {controlsOpen && (
                      <GenerationControls
                        sampling={sampling}
                        presets={generationPresets}
                        activePreset={activePreset}
                        showThinking={showThinking}
                        canThink={canThink}
                        onSamplingChange={(next) => {
                          setSampling(next);
                          setActivePreset(null);
                        }}
                        onPresetChange={applyPreset}
                        onThinkingChange={setShowThinking}
                        onReset={resetControls}
                      />
                    )}
                  </div>

                  {(activePreset || hasCustomSampling || showThinking) && (
                    <span className="ml-1 hidden max-w-[240px] truncate text-[11px] text-[var(--text-2)] sm:inline">
                      {activePresetConfig?.label ?? "Custom"}
                      {showThinking ? " / Thinking" : ""}
                    </span>
                  )}
                </div>

                {isStreaming ? (
                  <button
                    type="button"
                    onClick={onStop}
                    aria-label="Stop generating"
                    title="Stop generating"
                    className="flex h-8 w-8 items-center justify-center rounded-full bg-white text-black transition hover:bg-neutral-200"
                  >
                    <Square size={13} fill="currentColor" />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleSubmit}
                    disabled={!canSubmit}
                    aria-label={
                      primaryAction === "generate_image"
                        ? "Generate image with quality defaults"
                        : "Send message"
                    }
                    title={
                      sendDisabledReason ??
                      (primaryAction === "generate_image"
                        ? "Generate image with Q6 quality defaults"
                        : "Send message")
                    }
                    className="flex h-8 w-8 items-center justify-center rounded-full bg-white text-black transition hover:bg-neutral-200 disabled:bg-[#676767] disabled:text-[#303030]"
                  >
                    {primaryAction === "generate_image"
                      ? <ImageIcon size={16} />
                      : <ArrowUp size={17} strokeWidth={2.5} />}
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
      {canvasOpen && (
        <CanvasPanel
          versions={canvasVersions}
          index={canvasIndex}
          onSelect={setCanvasIndex}
          onClose={() => setCanvasOpen(false)}
        />
      )}
    </div>
  );
}

function ChatEmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex h-full items-center justify-center px-6 pb-16">
      <div className="text-center">
        {icon && (
          <div className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--surface-2)] text-[var(--text-1)]">
            {icon}
          </div>
        )}
        <h2 className="text-xl font-semibold text-[var(--text-0)]">{title}</h2>
        {description && <p className="mt-2 text-sm text-[var(--text-2)]">{description}</p>}
        {action && <div className="mt-5">{action}</div>}
      </div>
    </div>
  );
}

function ImageGenerationControls({
  presets,
  busy,
  requiresManualUnload,
  onGenerate,
  onClose,
}: {
  presets: ImageSizePreset[];
  busy: boolean;
  requiresManualUnload: boolean;
  onGenerate: (options: {
    size: ImageSizePreset;
    steps: number;
    cfgScale: number;
    sampler: string;
    seed: number | null;
    negativePrompt: string;
  }) => void;
  onClose: () => void;
}) {
  const defaultPreset =
    presets.find((preset) => preset.id === "recommended_square") ?? presets[0];
  const [sizeId, setSizeId] = useState(defaultPreset?.id ?? "");
  const [steps, setSteps] = useState(50);
  const [advanced, setAdvanced] = useState(false);
  const [randomSeed, setRandomSeed] = useState(true);
  const [seed, setSeed] = useState("42");
  const [cfgScale, setCfgScale] = useState(2.5);
  const [sampler, setSampler] = useState("euler");
  const [negativePrompt, setNegativePrompt] = useState("");
  const selected = presets.find((preset) => preset.id === sizeId) ?? defaultPreset;
  if (!selected) return null;

  return (
    <div className="absolute bottom-full left-0 z-50 mb-3 w-[480px] max-w-[calc(100vw-32px)] rounded-2xl border border-white/10 bg-[#292929] p-4 shadow-[0_18px_56px_rgba(0,0,0,0.5)]">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-sm font-semibold text-[var(--text-0)]">Generate a high-quality image</div>
          <div className="mt-1 text-[11px] leading-4 text-[var(--text-2)]">
            {requiresManualUnload
              ? "Automatic swapping is safety-locked. Unload the chat model first; this composer stays available for image generation."
              : "IB will render with Qwen-Image and keep the result attached to this chat."}
          </div>
        </div>
        <IconButton label="Close image options" size="sm" onClick={onClose}>
          <X size={14} />
        </IconButton>
      </div>

      <label className="mt-4 block">
        <span className="mb-1.5 block text-[11px] font-medium text-[var(--text-1)]">Size and shape</span>
        <select
          value={sizeId}
          onChange={(event) => setSizeId(event.target.value)}
          className="h-10 w-full rounded-xl border border-white/10 bg-black/15 px-3 text-sm text-white outline-none focus:border-white/25"
        >
          {presets.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.name} · {preset.aspect_ratio} · {preset.width}×{preset.height}
            </option>
          ))}
        </select>
      </label>
      <div className={`mt-2 rounded-lg px-3 py-2 text-[11px] leading-4 ${
        selected.tier === "max"
          ? "bg-amber-400/8 text-amber-200"
          : "bg-black/10 text-[var(--text-2)]"
      }`}>
        {selected.note}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3">
        <label>
          <span className="mb-1.5 block text-[11px] font-medium text-[var(--text-1)]">Quality steps</span>
          <select
            value={steps}
            onChange={(event) => setSteps(Number(event.target.value))}
            className="h-9 w-full rounded-lg border border-white/10 bg-black/15 px-2 text-xs text-white outline-none"
          >
            <option value={30}>30 · Faster</option>
            <option value={40}>40 · Balanced</option>
            <option value={50}>50 · Quality (recommended)</option>
            <option value={60}>60 · Extra refinement</option>
          </select>
        </label>
        <label>
          <span className="mb-1.5 block text-[11px] font-medium text-[var(--text-1)]">Seed</span>
          <div className="flex h-9 items-center gap-2 rounded-lg border border-white/10 bg-black/15 px-2">
            <input
              type="checkbox"
              checked={randomSeed}
              onChange={(event) => setRandomSeed(event.target.checked)}
              className="accent-violet-400"
            />
            {randomSeed ? (
              <span className="text-xs text-[var(--text-2)]">Random</span>
            ) : (
              <input
                type="number"
                value={seed}
                onChange={(event) => setSeed(event.target.value)}
                className="min-w-0 flex-1 bg-transparent text-xs text-white outline-none"
              />
            )}
          </div>
        </label>
      </div>

      <button
        type="button"
        onClick={() => setAdvanced((value) => !value)}
        className="mt-3 text-[11px] font-medium text-[var(--text-2)] hover:text-white"
      >
        {advanced ? "Hide advanced options" : "Show advanced options"}
      </button>
      {advanced && (
        <div className="mt-2 rounded-xl border border-white/8 bg-black/10 p-3">
          <div className="grid grid-cols-2 gap-3">
            <label>
              <span className="mb-1 block text-[10px] text-[var(--text-2)]">CFG scale</span>
              <input
                type="number"
                min={0}
                max={20}
                step={0.1}
                value={cfgScale}
                onChange={(event) => setCfgScale(Number(event.target.value))}
                className="h-8 w-full rounded-lg border border-white/10 bg-black/15 px-2 text-xs text-white outline-none"
              />
            </label>
            <label>
              <span className="mb-1 block text-[10px] text-[var(--text-2)]">Sampler</span>
              <select
                value={sampler}
                onChange={(event) => setSampler(event.target.value)}
                className="h-8 w-full rounded-lg border border-white/10 bg-black/15 px-2 text-xs text-white outline-none"
              >
                <option value="euler">Euler (tested)</option>
                <option value="euler_a">Euler A</option>
                <option value="heun">Heun</option>
                <option value="dpm2">DPM2</option>
                <option value="dpm++2m">DPM++ 2M</option>
              </select>
            </label>
          </div>
          <label className="mt-3 block">
            <span className="mb-1 block text-[10px] text-[var(--text-2)]">Negative prompt</span>
            <textarea
              value={negativePrompt}
              onChange={(event) => setNegativePrompt(event.target.value)}
              rows={2}
              placeholder="Optional things to avoid, e.g. blurry, distorted text, extra fingers"
              className="w-full resize-none rounded-lg border border-white/10 bg-black/15 px-2 py-1.5 text-xs text-white outline-none placeholder:text-[var(--text-3)]"
            />
          </label>
        </div>
      )}

      <div className="mt-4 flex items-center justify-between gap-3">
        <div className="text-[10px] text-[var(--text-3)]">
          Progress, elapsed time and ETA will appear in chat.
        </div>
        <Button
          variant="primary"
          disabled={busy || requiresManualUnload || (!randomSeed && !seed.trim())}
          onClick={() => onGenerate({
            size: selected,
            steps,
            cfgScale,
            sampler,
            seed: randomSeed ? null : Number(seed),
            negativePrompt,
          })}
        >
          {busy ? "Starting..." : requiresManualUnload ? "Unload model first" : "Generate image"}
        </Button>
      </div>
    </div>
  );
}

function ImageGenerationProgressCard({
  progress,
  receivedAt,
  onCancel,
}: {
  progress: ImageGenerationProgress;
  receivedAt: number;
  onCancel: () => void;
}) {
  const liveDelta = progress.done ? 0 : Math.max(0, (Date.now() - receivedAt) / 1_000);
  const elapsed = progress.elapsed_seconds + liveDelta;
  const eta = progress.eta_seconds == null
    ? null
    : Math.max(0, progress.eta_seconds - liveDelta);
  const percentage = Math.round(Math.max(0, Math.min(1, progress.progress)) * 100);
  const failed = progress.status === "failed";
  const cancelled = progress.status === "cancelled";
  const statusText = failed
    ? progress.error ?? progress.message
    : cancelled
      ? "Generation cancelled"
      : progress.message;
  const title = failed
    ? "Image generation failed"
    : cancelled
      ? "Image cancelled"
      : progress.done
        ? "Image ready"
        : "Generating image";

  return (
    <div
      className={`my-5 rounded-2xl border px-4 py-4 ${
        failed
          ? "border-rose-400/20 bg-rose-950/20"
          : "border-white/10 bg-[var(--surface-2)]"
      }`}
      aria-live="polite"
    >
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-400/10 text-violet-300">
          <ImageIcon size={16} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="text-sm font-semibold text-[var(--text-0)]">
                {title}
              </div>
              <div className={`mt-0.5 text-xs ${failed ? "text-rose-200" : "text-[var(--text-2)]"}`}>
                {statusText}
              </div>
            </div>
            {!progress.done && (
              <Button variant="ghost" size="sm" icon={<Square size={11} />} onClick={onCancel}>
                Cancel
              </Button>
            )}
          </div>

          <div
            className="mt-3 h-1.5 overflow-hidden rounded-full bg-black/25"
            role="progressbar"
            aria-label="Image generation progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percentage}
          >
            <div
              className="h-full rounded-full bg-violet-400 transition-[width] duration-300"
              style={{ width: `${percentage}%` }}
            />
          </div>

          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-[var(--text-2)]">
            <span>{percentage}% complete</span>
            {progress.total_steps > 0 && progress.stage === "generating" && (
              <span>Step {progress.current_step} of {progress.total_steps}</span>
            )}
            <span>Elapsed {formatImageDuration(elapsed)}</span>
            {!progress.done && (
              <span>
                {eta == null ? "Estimating time remaining..." : `About ${formatImageDuration(eta)} left`}
              </span>
            )}
          </div>
          {progress.output_path && (
            <div className="mt-2 truncate text-[11px] text-[var(--text-3)]" title={progress.output_path}>
              Saved to {progress.output_path}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function formatImageDuration(seconds: number) {
  const rounded = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(rounded / 60);
  const remainingSeconds = rounded % 60;
  if (minutes === 0) return `${remainingSeconds}s`;
  return `${minutes}m ${remainingSeconds.toString().padStart(2, "0")}s`;
}

function GenerationControls({
  sampling,
  presets,
  activePreset,
  showThinking,
  canThink,
  onSamplingChange,
  onPresetChange,
  onThinkingChange,
  onReset,
}: {
  sampling: SamplingParams;
  presets: Array<{
    key: string;
    label: string;
    description: string;
    sampling: SamplingParams;
    suggestThinking: boolean | null;
  }>;
  activePreset: string | null;
  showThinking: boolean;
  canThink: boolean;
  onSamplingChange: (sampling: SamplingParams) => void;
  onPresetChange: (preset: string) => void;
  onThinkingChange: (value: boolean) => void;
  onReset: () => void;
}) {
  const setNumber = (key: keyof SamplingParams, value: string, integer = false) => {
    onSamplingChange({
      ...sampling,
      [key]: value ? (integer ? Number.parseInt(value, 10) : Number.parseFloat(value)) : undefined,
    });
  };

  return (
    <div className="absolute bottom-full left-0 z-50 mb-3 w-[430px] max-w-[calc(100vw-32px)] rounded-2xl border border-white/10 bg-[#292929] p-3 shadow-[0_18px_56px_rgba(0,0,0,0.5)]">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold text-[var(--text-0)]">Generation controls</div>
          <div className="mt-0.5 text-[11px] text-[var(--text-2)]">Applied to the next message</div>
        </div>
        <Button variant="ghost" size="sm" icon={<RotateCcw size={13} />} onClick={onReset}>
          Reset
        </Button>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-1 rounded-xl bg-black/15 p-1">
        {presets.map((preset) => (
          <button
            key={preset.key}
            type="button"
            title={preset.description}
            onClick={() => onPresetChange(preset.key)}
            className={`h-8 rounded-lg text-xs font-medium transition ${
              activePreset === preset.key
                ? "bg-white text-black"
                : "text-[var(--text-1)] hover:bg-white/5 hover:text-white"
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <SamplingField
          label="Temperature"
          value={sampling.temperature}
          placeholder="0.7"
          min="0"
          max="2"
          step="0.1"
          onChange={(value) => setNumber("temperature", value)}
        />
        <SamplingField
          label="Top P"
          value={sampling.top_p}
          placeholder="0.9"
          min="0"
          max="1"
          step="0.05"
          onChange={(value) => setNumber("top_p", value)}
        />
        <SamplingField
          label="Top K"
          value={sampling.top_k}
          placeholder="40"
          min="0"
          step="1"
          onChange={(value) => setNumber("top_k", value, true)}
        />
        <SamplingField
          label="Min P"
          value={sampling.min_p}
          placeholder="0"
          min="0"
          max="1"
          step="0.01"
          onChange={(value) => setNumber("min_p", value)}
        />
        <SamplingField
          label="Presence"
          value={sampling.presence_penalty}
          placeholder="0"
          min="-2"
          max="2"
          step="0.1"
          onChange={(value) => setNumber("presence_penalty", value)}
        />
        <SamplingField
          label="Repeat"
          value={sampling.repeat_penalty}
          placeholder="1"
          min="0"
          max="2"
          step="0.05"
          onChange={(value) => setNumber("repeat_penalty", value)}
        />
        <SamplingField
          label="Max tokens"
          value={sampling.max_tokens}
          placeholder="2048"
          min="1"
          step="64"
          onChange={(value) => setNumber("max_tokens", value, true)}
        />
        <SamplingField
          label="Seed"
          value={sampling.seed}
          placeholder="-1"
          step="1"
          onChange={(value) => setNumber("seed", value, true)}
        />
      </div>

      <div className="mt-3 flex items-center justify-between rounded-xl border border-white/8 bg-black/10 px-3 py-2.5">
        <div>
          <div className="text-xs font-medium text-[var(--text-0)]">Show reasoning</div>
          <div className="mt-0.5 text-[11px] text-[var(--text-2)]">
            {canThink ? "Display reasoning emitted by the loaded mode; changing this does not change --reasoning" : "The loaded model does not expose reasoning output"}
          </div>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={showThinking}
          aria-label="Show reasoning output"
          disabled={!canThink}
          onClick={() => canThink && onThinkingChange(!showThinking)}
          className={`relative h-6 w-11 rounded-full transition disabled:cursor-not-allowed disabled:opacity-40 ${showThinking ? "bg-white" : "bg-[#555]"}`}
        >
          <span
            className={`absolute top-1 h-4 w-4 rounded-full transition-all ${
              showThinking ? "left-6 bg-black" : "left-1 bg-white"
            }`}
          />
        </button>
      </div>
    </div>
  );
}

function SamplingField({
  label,
  value,
  placeholder,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value?: number;
  placeholder: string;
  min?: string;
  max?: string;
  step: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="min-w-0">
      <span className="mb-1 block truncate text-[10px] text-[var(--text-2)]" title={label}>
        {label}
      </span>
      <input
        type="number"
        value={value ?? ""}
        placeholder={placeholder}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(event.target.value)}
        className="h-8 w-full rounded-lg border border-white/10 bg-black/15 px-2 text-xs text-white outline-none placeholder:text-[var(--text-3)] focus:border-white/25"
      />
    </label>
  );
}

import { useState, useRef, useEffect } from "react";
import type { MessageInfo } from "../../lib/types";
import type { SamplingParams } from "../../lib/tauri";
import { PRESETS, PRESET_ORDER, modelSupportsThinking, type PresetKey } from "../../lib/presets";
import { MessageBubble } from "./MessageBubble";
import { StreamingText } from "./StreamingText";

interface Props {
  messages: MessageInfo[];
  isStreaming: boolean;
  streamingText: string;
  streamingReasoning: string;
  tokensPerSecond: number | null;
  error: string | null;
  hasModel: boolean;
  hasSession: boolean;
  /** Name of the currently loaded model used to detect thinking support. */
  loadedModel?: string | null;
  loadedModelSupportsVision?: boolean;
  loadedModelVisionStatusText?: string | null;
  onSend: (
    content: string,
    sampling?: SamplingParams,
    imageBase64?: string | null,
    showThinking?: boolean | null
  ) => void;
  onStop: () => void;
}

export function ChatPanel({
  messages,
  isStreaming,
  streamingText,
  streamingReasoning,
  tokensPerSecond,
  error,
  hasModel,
  hasSession,
  loadedModel,
  loadedModelSupportsVision = false,
  loadedModelVisionStatusText = null,
  onSend,
  onStop,
}: Props) {
  const [input, setInput] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [showSampling, setShowSampling] = useState(false);
  const [sampling, setSampling] = useState<SamplingParams>({});
  const [showThinking, setShowThinking] = useState(false);
  const [activePreset, setActivePreset] = useState<PresetKey | null>(null);
  const [composerError, setComposerError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const canThink = modelSupportsThinking(loadedModel);

  const applyPreset = (key: PresetKey) => {
    const preset = PRESETS[key];
    if (!preset) return;
    setActivePreset(key);
    setSampling(preset.sampling);
    if (preset.suggestThinking !== null && canThink) {
      setShowThinking(preset.suggestThinking);
    }
  };

  const clearPreset = () => {
    setActivePreset(null);
    setSampling({});
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setImage(file);
    setComposerError(null);
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setImagePreview(ev.target?.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }
  };

  const handleRemoveImage = () => {
    setImage(null);
    setImagePreview(null);
    setComposerError(null);
  };

  // Allow pasting images directly from the clipboard into the chat input.
  const handlePaste = (e: React.ClipboardEvent) => {
    const items = Array.from(e.clipboardData?.items ?? []);
    const imageItem = items.find((item) => item.type.startsWith("image/"));
    if (!imageItem) return;
    e.preventDefault();
    const file = imageItem.getAsFile();
    if (!file) return;
    setImage(file);
    setComposerError(null);
    const reader = new FileReader();
    reader.onload = (ev) => setImagePreview(ev.target?.result as string);
    reader.readAsDataURL(file);
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, streamingText]);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if ((!trimmed && !image) || !hasModel || !hasSession || isStreaming) return;
    if (image && !loadedModelSupportsVision) {
      setComposerError(
        loadedModelVisionStatusText ??
          "The current model is not vision-ready. Load a vision model with its matching mmproj sidecar first."
      );
      return;
    }
    const params = Object.keys(sampling).length > 0 ? sampling : undefined;
    setComposerError(null);
    onSend(trimmed, params, imagePreview, showThinking);
    setInput("");
    setImage(null);
    setImagePreview(null);
  };

  if (!hasModel) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <p className="text-2xl mb-2">Chat</p>
          <p>Load a model in the Models tab to start chatting</p>
        </div>
      </div>
    );
  }

  if (!hasSession) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <p className="text-2xl mb-2">Chat</p>
          <p>Create or select a session from the sidebar</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && !isStreaming && (
          <div className="flex items-center justify-center h-full text-gray-600 text-sm">
            Start the conversation by sending a message
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isStreaming && <StreamingText text={streamingText} reasoning={streamingReasoning} />}
        <div ref={bottomRef} />
      </div>

      {(tokensPerSecond != null || error || composerError) && (
        <div className="px-4 py-1 text-xs border-t border-gray-700/50">
          {error || composerError ? (
            <span className="text-red-400">{composerError ?? error}</span>
          ) : (
            <span className="text-gray-500">
              {tokensPerSecond?.toFixed(1)} tok/s
            </span>
          )}
        </div>
      )}

      <div className="border-t border-gray-700 p-3">
        {/* Preset row */}
        <div className="flex items-center gap-1.5 mb-2 flex-wrap">
          <span className="text-[10px] uppercase tracking-widest text-gray-600 mr-0.5">Preset</span>
          {PRESET_ORDER.map((key) => {
            const p = PRESETS[key];
            const isActive = activePreset === key;
            return (
              <button
                key={key}
                onClick={() => (isActive ? clearPreset() : applyPreset(key))}
                title={p.description}
                className="flex items-center gap-1 rounded px-2 py-0.5 text-xs transition"
                style={{
                  background: isActive ? "rgba(34,211,238,0.14)" : "rgba(255,255,255,0.04)",
                  border: isActive
                    ? "1px solid rgba(34,211,238,0.35)"
                    : "1px solid rgba(255,255,255,0.08)",
                  color: isActive ? "#22d3ee" : "#6b7280",
                  cursor: "pointer",
                  fontWeight: isActive ? 500 : 400,
                }}
              >
                <span>{p.icon}</span>
                {p.label}
              </button>
            );
          })}

          <div style={{ width: "1px", height: "14px", background: "rgba(255,255,255,0.08)", margin: "0 2px" }} />

          {/* Thinking toggle only shown for models that support it */}
          {canThink && (
            <button
              onClick={() => setShowThinking(!showThinking)}
              title={
                showThinking
                  ? "Thinking ON - model reasons step-by-step before answering"
                  : "Thinking OFF - direct response (faster)"
              }
              className="flex items-center gap-1.5 rounded px-2 py-0.5 text-xs transition"
              style={{
                background: showThinking ? "rgba(167,139,250,0.12)" : "transparent",
                border: showThinking ? "1px solid rgba(167,139,250,0.35)" : "1px solid rgba(255,255,255,0.08)",
                color: showThinking ? "#a78bfa" : "#52525a",
                cursor: "pointer",
              }}
            >
              Think
              <span
                className="relative shrink-0 rounded-full transition"
                style={{
                  display: "inline-block",
                  width: "24px",
                  height: "13px",
                  background: showThinking ? "#a78bfa" : "rgba(255,255,255,0.1)",
                  verticalAlign: "middle",
                }}
              >
                <span
                  className="absolute rounded-full bg-white transition-all"
                  style={{
                    width: "9px", height: "9px",
                    top: "2px",
                    left: showThinking ? "13px" : "2px",
                  }}
                />
              </span>
            </button>
          )}

          {/* Show thinking for non-thinking models too (manual override) */}
          {!canThink && (
            <button
              onClick={() => setShowThinking(!showThinking)}
              title="Thinking toggle (model may not support this)"
              className="flex items-center gap-1 rounded px-2 py-0.5 text-xs transition"
              style={{
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#3f3f46",
                cursor: "pointer",
                background: "transparent",
              }}
            >
              Think
            </button>
          )}

          {/* Sampling custom button shows or hides the fine-tune panel */}
          <button
            onClick={() => setShowSampling(!showSampling)}
            className="flex items-center gap-1 rounded px-2 py-0.5 text-xs transition"
            style={{
              border: "1px solid rgba(255,255,255,0.08)",
              color: Object.values(sampling).some((v) => v !== undefined) ? "#22d3ee" : "#4b5563",
              background: Object.values(sampling).some((v) => v !== undefined) ? "rgba(34,211,238,0.06)" : "transparent",
              cursor: "pointer",
            }}
            title="Fine-tune individual sampling parameters"
          >
            {showSampling ? "Hide Params" : "Params"}
            {Object.values(sampling).some((v) => v !== undefined) && !activePreset && (
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: "#22d3ee" }} />
            )}
          </button>

          {(activePreset || Object.values(sampling).some((v) => v !== undefined)) && (
            <button
              onClick={clearPreset}
              className="rounded px-2 py-0.5 text-xs transition"
              style={{ color: "#6b7280", border: "none", background: "transparent", cursor: "pointer" }}
              title="Reset to model defaults"
            >
              Reset
            </button>
          )}
        </div>

        {showSampling && (
          <div className="flex flex-wrap gap-3 mb-2 text-xs">
            <label className="flex items-center gap-1.5">
              <span className="text-gray-500">Temp</span>
              <input
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={sampling.temperature ?? ""}
                onChange={(e) =>
                  setSampling({
                    ...sampling,
                    temperature: e.target.value
                      ? parseFloat(e.target.value)
                      : undefined,
                  })
                }
                placeholder="0.7"
                className="w-16 px-1.5 py-0.5 bg-gray-800 border border-gray-600 rounded text-gray-300 focus:outline-none focus:border-blue-500"
              />
            </label>
            <label className="flex items-center gap-1.5">
              <span className="text-gray-500">Top P</span>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={sampling.top_p ?? ""}
                onChange={(e) =>
                  setSampling({
                    ...sampling,
                    top_p: e.target.value ? parseFloat(e.target.value) : undefined,
                  })
                }
                placeholder="0.9"
                className="w-16 px-1.5 py-0.5 bg-gray-800 border border-gray-600 rounded text-gray-300 focus:outline-none focus:border-blue-500"
              />
            </label>
            <label className="flex items-center gap-1.5">
              <span className="text-gray-500">Top K</span>
              <input
                type="number"
                step="1"
                min="0"
                value={sampling.top_k ?? ""}
                onChange={(e) =>
                  setSampling({
                    ...sampling,
                    top_k: e.target.value ? parseInt(e.target.value) : undefined,
                  })
                }
                placeholder="40"
                className="w-16 px-1.5 py-0.5 bg-gray-800 border border-gray-600 rounded text-gray-300 focus:outline-none focus:border-blue-500"
              />
            </label>
            <label className="flex items-center gap-1.5">
              <span className="text-gray-500">Max</span>
              <input
                type="number"
                step="64"
                min="1"
                value={sampling.max_tokens ?? ""}
                onChange={(e) =>
                  setSampling({
                    ...sampling,
                    max_tokens: e.target.value
                      ? parseInt(e.target.value)
                      : undefined,
                  })
                }
                placeholder="2048"
                className="w-20 px-1.5 py-0.5 bg-gray-800 border border-gray-600 rounded text-gray-300 focus:outline-none focus:border-blue-500"
              />
            </label>
            <label className="flex items-center gap-1.5">
              <span className="text-gray-500">Seed</span>
              <input
                type="number"
                step="1"
                value={sampling.seed ?? ""}
                onChange={(e) =>
                  setSampling({
                    ...sampling,
                    seed: e.target.value ? parseInt(e.target.value) : undefined,
                  })
                }
                placeholder="-1"
                className="w-20 px-1.5 py-0.5 bg-gray-800 border border-gray-600 rounded text-gray-300 focus:outline-none focus:border-blue-500"
              />
            </label>
          </div>
        )}

        <div className="flex gap-2 items-end">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
              if (e.key === "Escape" && isStreaming) {
                e.preventDefault();
                onStop();
              }
            }}
            onPaste={handlePaste}
            placeholder="Type a message... (Enter to send, Shift+Enter for newline, paste image)"
            rows={1}
            className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500 resize-none focus:outline-none focus:border-blue-500"
          />
          <label className="cursor-pointer flex items-center">
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleImageChange}
              disabled={isStreaming}
            />
            <span className="px-2 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm text-gray-200 ml-1">
              Img
            </span>
          </label>
          {isStreaming ? (
            <button
              onClick={onStop}
              className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded-lg text-sm shrink-0"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!input.trim() && !image}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm disabled:opacity-40 shrink-0"
            >
              Send
            </button>
          )}
        </div>
        {imagePreview && (
          <div className="mt-2 flex items-center gap-2">
            <img
              src={imagePreview}
              alt="preview"
              className="max-h-24 rounded border border-gray-600"
            />
            <span className="text-xs text-gray-500">
              {loadedModelSupportsVision
                ? "Will be sent to the current vision model."
                : loadedModelVisionStatusText ??
                  "Current model is text-only. Load a vision model before sending."}
            </span>
            <button
              onClick={handleRemoveImage}
              className="text-xs text-red-400 hover:underline"
            >
              Remove
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

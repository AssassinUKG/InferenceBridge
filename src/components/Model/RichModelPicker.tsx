import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  HardDrive,
  Library,
  LoaderCircle,
  RefreshCw,
  Search,
  Server,
  X,
} from "lucide-react";
import type { LoadProgress, ModelInfo, ProcessStatusInfo } from "../../lib/types";
import { Button } from "../ui/Controls";
import {
  CapabilityBadge,
  ModelArtwork,
  modelContextLabel,
  modelDisplayName,
  modelParameterLabel,
  modelPresentationKey,
  modelPublisher,
  modelSummary,
} from "./modelPresentation";

type CapabilityFilter = "all" | "tools" | "reasoning" | "vision";
type SortKey = "name" | "size-asc" | "size-desc" | "parameters";

interface Props {
  open: boolean;
  models: ModelInfo[];
  loadedModel: string | null;
  processStatus: ProcessStatusInfo | null;
  isLoading: boolean;
  loadProgress: LoadProgress | null;
  error: string | null;
  returnFocus: HTMLElement | null;
  switchingDisabledReason?: string | null;
  onClose: () => void;
  onConfigureLoad: (model: ModelInfo) => void;
  onOpenLibrary: () => void;
  onScan: () => void;
}

const FILTERS: Array<{ key: CapabilityFilter; label: string }> = [
  { key: "all", label: "All" },
  { key: "tools", label: "Tools" },
  { key: "reasoning", label: "Reasoning" },
  { key: "vision", label: "Vision" },
];

function parameterValue(model: ModelInfo) {
  const label = modelParameterLabel(model);
  if (!label) return -1;
  const value = Number.parseFloat(label);
  if (!Number.isFinite(value)) return -1;
  return label.endsWith("M") ? value / 1000 : value;
}

function formatSize(model: ModelInfo, precision = 1) {
  return model.size_gb > 0 ? `${model.size_gb.toFixed(precision)} GB` : "Unknown";
}

function sortModels(models: ModelInfo[], sort: SortKey, loadedModel: string | null) {
  return [...models].sort((a, b) => {
    const aLoaded = a.filename === loadedModel;
    const bLoaded = b.filename === loadedModel;
    if (aLoaded !== bLoaded) return aLoaded ? -1 : 1;

    if (sort === "size-asc") return (a.size_gb || Number.POSITIVE_INFINITY) - (b.size_gb || Number.POSITIVE_INFINITY);
    if (sort === "size-desc") return (b.size_gb || -1) - (a.size_gb || -1);
    if (sort === "parameters") return parameterValue(b) - parameterValue(a);
    return modelDisplayName(a).localeCompare(modelDisplayName(b), undefined, { numeric: true, sensitivity: "base" });
  });
}

function matchesFilter(model: ModelInfo, filter: CapabilityFilter) {
  if (filter === "tools") return model.supports_tools;
  if (filter === "reasoning") return model.supports_reasoning;
  if (filter === "vision") return model.supports_vision;
  return true;
}

function searchText(model: ModelInfo) {
  return [
    model.filename,
    model.hf_repo,
    modelDisplayName(model),
    model.family,
    model.gguf_architecture,
    modelPublisher(model),
    model.provider_name,
    model.quant,
  ].filter(Boolean).join(" ").toLowerCase();
}

function loadActionLabel(model: ModelInfo, loadedModel: string | null) {
  if (model.filename === loadedModel) return "Reload options";
  if (loadedModel) return "Switch options";
  return "Load options";
}

function PickerMeta({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex h-5 items-center rounded-md px-1.5 font-mono text-[9px]" style={{ background: "rgba(255,255,255,0.055)", border: "1px solid var(--border)", color: "var(--text-1)" }}>
      {children}
    </span>
  );
}

function DetailMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-xl px-3 py-3" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
      <div className="text-[9px] font-semibold uppercase tracking-[0.14em]" style={{ color: "var(--text-2)" }}>{label}</div>
      <div className="mt-1 truncate text-xs font-semibold" style={{ color: "var(--text-0)" }} title={value}>{value}</div>
    </div>
  );
}

export function RichModelPicker({
  open,
  models,
  loadedModel,
  processStatus,
  isLoading,
  loadProgress,
  error,
  returnFocus,
  switchingDisabledReason = null,
  onClose,
  onConfigureLoad,
  onOpenLibrary,
  onScan,
}: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<CapabilityFilter>("all");
  const [sort, setSort] = useState<SortKey>("name");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [mobileDetailOpen, setMobileDetailOpen] = useState(false);
  const modelFingerprint = models.map((model) => modelPresentationKey(model)).join("\n");

  const visibleModels = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const allowed = models.filter((model) => {
      const isActiveExternal = model.filename === loadedModel && !model.provider_managed;
      if (!model.provider_managed && !isActiveExternal) return false;
      if (!matchesFilter(model, filter)) return false;
      return !normalizedQuery || searchText(model).includes(normalizedQuery);
    });
    return sortModels(allowed, sort, loadedModel);
  }, [filter, loadedModel, modelFingerprint, models, query, sort]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedModel = visibleModels.find((model) => modelPresentationKey(model) === selectedKey)
    ?? visibleModels.find((model) => model.filename === loadedModel)
    ?? visibleModels[0]
    ?? null;

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      dialog.showModal();
      window.requestAnimationFrame(() => searchRef.current?.focus());
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const preferred = models.find((model) => model.filename === loadedModel) ?? models.find((model) => model.provider_managed) ?? null;
    setQuery("");
    setFilter("all");
    setSelectedKey(preferred ? modelPresentationKey(preferred) : null);
    setMobileDetailOpen(false);
  }, [loadedModel, modelFingerprint, open]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!open) return undefined;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      onClose();
    };
    window.addEventListener("keydown", closeOnEscape, true);
    return () => window.removeEventListener("keydown", closeOnEscape, true);
  }, [onClose, open]);

  useEffect(() => {
    if (!selectedModel) {
      setSelectedKey(null);
      return;
    }
    if (selectedKey !== modelPresentationKey(selectedModel)) {
      setSelectedKey(modelPresentationKey(selectedModel));
    }
  }, [selectedKey, selectedModel]);

  useEffect(() => {
    return () => {
      const dialog = dialogRef.current;
      if (dialog?.open) dialog.close();
    };
  }, []);

  const localModels = models.filter((model) => model.provider_managed);
  const activeExternalPresent = models.some((model) => model.filename === loadedModel && !model.provider_managed);
  const eligibleModelCount = localModels.length + (activeExternalPresent ? 1 : 0);
  const localDiskGb = localModels.reduce((sum, model) => sum + Math.max(0, model.size_gb || 0), 0);
  const selectedLoaded = selectedModel?.filename === loadedModel;
  const selectedExternal = selectedModel ? !selectedModel.provider_managed : false;
  const loadDisabledReason = selectedExternal
    ? "External runtimes are selected through their provider settings."
    : switchingDisabledReason || (isLoading ? "A model transition is already in progress." : null);
  const liveContext = selectedLoaded ? processStatus?.last_launch_preview?.context_size ?? null : null;
  const contextLabel = selectedModel ? modelContextLabel(selectedModel) : null;
  const activeProgress = loadProgress && !loadProgress.done ? loadProgress : null;

  const moveSelection = (direction: -1 | 1 | "first" | "last", focusOption = false) => {
    if (visibleModels.length === 0) return;
    const currentIndex = selectedModel
      ? visibleModels.findIndex((model) => modelPresentationKey(model) === modelPresentationKey(selectedModel))
      : -1;
    const nextIndex = direction === "first"
      ? 0
      : direction === "last"
        ? visibleModels.length - 1
        : Math.max(0, Math.min(visibleModels.length - 1, currentIndex + direction));
    const next = visibleModels[nextIndex];
    setSelectedKey(modelPresentationKey(next));
    const option = document.getElementById(`model-picker-option-${nextIndex}`);
    option?.scrollIntoView({ block: "nearest" });
    if (focusOption) window.requestAnimationFrame(() => option?.focus());
  };

  const handleListNavigation = (event: React.KeyboardEvent) => {
    const focusOption = event.currentTarget !== searchRef.current;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveSelection(1, focusOption);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      moveSelection(-1, focusOption);
    } else if (event.key === "Home") {
      event.preventDefault();
      moveSelection("first", focusOption);
    } else if (event.key === "End") {
      event.preventDefault();
      moveSelection("last", focusOption);
    } else if (event.key === "Enter" && event.currentTarget === searchRef.current && selectedModel) {
      event.preventDefault();
      if (!loadDisabledReason) onConfigureLoad(selectedModel);
    }
  };

  return (
    <dialog
      ref={dialogRef}
      id="rich-model-picker"
      className="ib-model-picker-dialog m-auto h-[min(680px,calc(100vh-24px))] w-[min(980px,calc(100vw-24px))] max-w-none overflow-hidden p-0"
      aria-labelledby="rich-model-picker-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClose={() => {
        if (open) onClose();
        returnFocus?.focus();
      }}
      onMouseDown={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        const outside = event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom;
        if (outside) onClose();
      }}
    >
      <div className="grid h-full min-h-0 grid-rows-[56px_minmax(0,1fr)]" style={{ background: "var(--surface-0)" }}>
        <header className="flex items-center gap-3 border-b px-4" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
          <div className="min-w-0 flex-1">
            <h2 id="rich-model-picker-title" className="truncate text-sm font-semibold" style={{ color: "var(--text-0)" }}>Choose a local model</h2>
            <p className="mt-0.5 text-[10px]" style={{ color: "var(--text-2)" }}>The active runtime is shared by Chat, the API, and connected clients.</p>
          </div>
          {activeProgress && (
            <div className="hidden items-center gap-2 rounded-full px-2.5 py-1 text-[10px] sm:flex" style={{ background: "rgba(59,130,246,0.1)", color: "#93c5fd", border: "1px solid rgba(59,130,246,0.22)" }}>
              <LoaderCircle size={11} className="animate-spin" />
              <span className="max-w-48 truncate">{activeProgress.message}</span>
            </div>
          )}
          <button type="button" onClick={onClose} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition hover:bg-white/5" aria-label="Close model picker" style={{ color: "var(--text-1)" }}>
            <X size={16} />
          </button>
        </header>

        <div className="grid min-h-0 md:grid-cols-[minmax(285px,0.78fr)_minmax(0,1.22fr)]">
          <section className={`${mobileDetailOpen ? "hidden" : "grid"} min-h-0 grid-rows-[auto_auto_minmax(0,1fr)_44px] border-r md:grid`} style={{ borderColor: "var(--border)" }} aria-label="Local model results">
            <div className="px-3 pb-2 pt-3">
              <label className="relative block">
                <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2" style={{ color: "var(--text-2)" }} />
                <input
                  ref={searchRef}
                  type="search"
                  role="combobox"
                  aria-label="Search local models"
                  aria-expanded={open}
                  aria-controls="model-picker-results"
                  aria-activedescendant={selectedModel ? `model-picker-option-${visibleModels.findIndex((model) => modelPresentationKey(model) === modelPresentationKey(selectedModel))}` : undefined}
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={handleListNavigation}
                  placeholder="Search models, families, or authors…"
                  className="ib-field h-9 w-full pl-8 pr-3 text-xs"
                />
              </label>
            </div>

            <div className="flex items-center gap-1 overflow-x-auto border-b px-3 pb-2" style={{ borderColor: "var(--border)" }}>
              {FILTERS.map((item) => (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => setFilter(item.key)}
                  aria-pressed={filter === item.key}
                  className="h-7 shrink-0 rounded-md px-2.5 text-[10px] font-semibold transition"
                  style={{
                    background: filter === item.key ? "var(--surface-3)" : "transparent",
                    border: `1px solid ${filter === item.key ? "var(--border-mid)" : "transparent"}`,
                    color: filter === item.key ? "var(--text-0)" : "var(--text-2)",
                  }}
                >
                  {item.label}
                </button>
              ))}
              <label className="ml-auto flex shrink-0 items-center gap-1.5 text-[9px]" style={{ color: "var(--text-2)" }}>
                Sort
                <span className="relative">
                  <select value={sort} onChange={(event) => setSort(event.target.value as SortKey)} className="h-7 appearance-none rounded-md py-0 pl-2 pr-6 text-[10px] outline-none" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}>
                    <option value="name">Name</option>
                    <option value="size-asc">Smallest</option>
                    <option value="size-desc">Largest</option>
                    <option value="parameters">Parameters</option>
                  </select>
                  <ChevronDown size={11} className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2" />
                </span>
              </label>
            </div>

            <div ref={listRef} id="model-picker-results" role="listbox" aria-label="Available local models" className="min-h-0 overflow-y-auto px-2 py-2" onKeyDown={handleListNavigation}>
              {error && (
                <div className="mx-1 mb-2 rounded-lg px-2.5 py-2 text-[10px] leading-4" role="alert" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#fca5a5" }}>
                  {error}
                </div>
              )}
              {visibleModels.length === 0 ? (
                <div className="flex h-full min-h-44 flex-col items-center justify-center px-5 text-center">
                  <HardDrive size={20} style={{ color: "var(--text-3)" }} />
                  <div className="mt-3 text-xs font-semibold" style={{ color: "var(--text-0)" }}>{eligibleModelCount === 0 ? "No local models found" : "No matching models"}</div>
                  <p className="mt-1 max-w-56 text-[10px] leading-4" style={{ color: "var(--text-2)" }}>{eligibleModelCount === 0 ? "Scan your configured folders or open the model library to get started." : "Try a different search or capability filter."}</p>
                  <div className="mt-3 flex gap-2">
                    {eligibleModelCount === 0 ? (
                      <Button size="sm" variant="secondary" onClick={onScan} icon={<RefreshCw size={12} />}>Scan</Button>
                    ) : (
                      <Button size="sm" variant="secondary" onClick={() => { setQuery(""); setFilter("all"); }}>Clear filters</Button>
                    )}
                  </div>
                </div>
              ) : (
                visibleModels.map((model, index) => {
                  const selected = selectedModel ? modelPresentationKey(model) === modelPresentationKey(selectedModel) : false;
                  const loaded = model.filename === loadedModel;
                  const params = modelParameterLabel(model);
                  const context = modelContextLabel(model);
                  return (
                    <button
                      key={modelPresentationKey(model)}
                      id={`model-picker-option-${index}`}
                      type="button"
                      role="option"
                      aria-selected={selected}
                      aria-label={`${modelDisplayName(model)}, ${modelPublisher(model)}, ${model.quant ?? "unquantized"}, ${formatSize(model)}${loaded ? ", active" : ""}`}
                      onClick={() => {
                        setSelectedKey(modelPresentationKey(model));
                        if (window.matchMedia("(max-width: 767px)").matches) setMobileDetailOpen(true);
                      }}
                      onFocus={() => setSelectedKey(modelPresentationKey(model))}
                      className="mb-1 w-full rounded-xl p-2.5 text-left outline-none transition last:mb-0 focus-visible:ring-1 focus-visible:ring-white/50"
                      style={{
                        background: selected ? "rgba(99,102,241,0.18)" : loaded ? "rgba(52,211,153,0.055)" : "transparent",
                        border: `1px solid ${selected ? "rgba(129,140,248,0.42)" : loaded ? "rgba(52,211,153,0.16)" : "transparent"}`,
                      }}
                    >
                      <div className="flex items-start gap-2.5">
                        <ModelArtwork model={model} size="md" />
                        <div className="min-w-0 flex-1">
                          <div className="flex min-w-0 items-center gap-1.5">
                            <span className="truncate text-xs font-semibold" style={{ color: "var(--text-0)" }} title={modelDisplayName(model)}>{modelDisplayName(model)}</span>
                            {loaded && <Check size={12} className="shrink-0 text-emerald-400" aria-hidden="true" />}
                            <span className="ml-auto shrink-0 text-[9px] font-semibold" style={{ color: loaded ? "#6ee7b7" : "var(--text-2)" }}>{loaded ? "Active" : selected ? "Selected" : ""}</span>
                          </div>
                          <div className="mt-0.5 truncate text-[10px]" style={{ color: "var(--text-2)" }} title={model.hf_repo ?? model.filename}>{modelPublisher(model)} · {model.filename}</div>
                          <div className="mt-1.5 flex min-w-0 items-center gap-1 overflow-hidden">
                            {model.quant && <PickerMeta>{model.quant}</PickerMeta>}
                            {params && <PickerMeta>{params}</PickerMeta>}
                            {model.size_gb > 0 && <PickerMeta>{formatSize(model)}</PickerMeta>}
                            {context && <PickerMeta>{context} ctx</PickerMeta>}
                            <span className="ml-auto flex shrink-0 gap-1">
                              {model.supports_vision && <CapabilityBadge capability="vision" ready={loaded && model.vision_runtime_ready} compact />}
                              {model.supports_tools && <CapabilityBadge capability="tools" compact />}
                              {model.supports_reasoning && <CapabilityBadge capability="reasoning" compact />}
                            </span>
                          </div>
                        </div>
                      </div>
                    </button>
                  );
                })
              )}
            </div>

            <footer className="flex items-center gap-2 border-t px-3 text-[10px]" style={{ borderColor: "var(--border)", color: "var(--text-2)", background: "var(--surface-1)" }}>
              <HardDrive size={11} />
              <span>{localModels.length} local · {localDiskGb.toFixed(1)} GB</span>
              <button type="button" onClick={onOpenLibrary} className="ml-auto flex items-center gap-1 font-semibold transition hover:text-white" style={{ color: "var(--text-1)" }}>
                <Library size={11} /> Manage models
              </button>
            </footer>
          </section>

          <aside className={`${mobileDetailOpen ? "grid" : "hidden"} min-h-0 grid-rows-[minmax(0,1fr)_auto] md:grid`} style={{ background: "var(--bg)" }} aria-label="Selected model details">
            {selectedModel ? (
              <div className="min-h-0 overflow-y-auto px-4 py-4 sm:px-5">
                <button type="button" onClick={() => setMobileDetailOpen(false)} className="mb-3 flex items-center gap-1 text-xs md:hidden" style={{ color: "var(--text-1)" }}><ArrowLeft size={14} />Back to models</button>

                <div className="flex items-start gap-3">
                  <ModelArtwork model={selectedModel} size="lg" />
                  <div className="min-w-0 flex-1">
                    <div className="flex min-w-0 items-start gap-2">
                      <div className="min-w-0 flex-1">
                        <h3 className="truncate text-base font-semibold" style={{ color: "var(--text-0)" }} title={modelDisplayName(selectedModel)}>{modelDisplayName(selectedModel)}</h3>
                        <p className="mt-0.5 truncate text-[10px]" style={{ color: "var(--text-2)" }} title={selectedModel.hf_repo ?? selectedModel.path}>{selectedModel.hf_repo ?? selectedModel.filename}</p>
                      </div>
                      {selectedLoaded && <span className="flex shrink-0 items-center gap-1 rounded-full px-2 py-1 text-[9px] font-bold uppercase tracking-wider" style={{ background: "rgba(52,211,153,0.1)", border: "1px solid rgba(52,211,153,0.24)", color: "#6ee7b7" }}><Check size={10} />Active</span>}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {selectedModel.supports_vision && <CapabilityBadge capability="vision" ready={selectedLoaded && selectedModel.vision_runtime_ready} />}
                      {selectedModel.supports_tools && <CapabilityBadge capability="tools" />}
                      {selectedModel.supports_reasoning && <CapabilityBadge capability="reasoning" />}
                      {!selectedModel.supports_vision && !selectedModel.supports_tools && !selectedModel.supports_reasoning && <span className="inline-flex h-6 items-center rounded-full px-2 text-[10px]" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}>Chat</span>}
                    </div>
                  </div>
                </div>

                <section className="mt-4 rounded-xl px-3 py-3" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
                  <div className="text-[9px] font-semibold uppercase tracking-[0.15em]" style={{ color: "var(--text-2)" }}>About this model</div>
                  <p className="mt-2 text-xs leading-5" style={{ color: "var(--text-1)" }}>{modelSummary(selectedModel)}</p>
                </section>

                <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4 md:grid-cols-2 lg:grid-cols-4">
                  <DetailMetric label="Size" value={formatSize(selectedModel, 2)} />
                  <DetailMetric label="Quant" value={selectedModel.quant ?? "Unquantized"} />
                  <DetailMetric label="Parameters" value={modelParameterLabel(selectedModel) ?? "Unknown"} />
                  <DetailMetric label="Context" value={liveContext ? `${liveContext.toLocaleString()} live` : contextLabel ? `${contextLabel} tokens` : "Unknown"} />
                </div>

                <section className="mt-3 overflow-hidden rounded-xl" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
                  {[
                    ["Publisher", modelPublisher(selectedModel)],
                    ["Architecture", selectedModel.family || selectedModel.gguf_architecture || "Unknown"],
                    ["Runtime", selectedModel.provider_name || (selectedModel.provider_managed ? "Managed llama.cpp" : "External")],
                    ["Model file", selectedModel.filename],
                  ].map(([label, value]) => (
                    <div key={label} className="grid grid-cols-[90px_minmax(0,1fr)] gap-3 border-b px-3 py-2.5 text-[10px] last:border-b-0" style={{ borderColor: "var(--border)" }}>
                      <span style={{ color: "var(--text-2)" }}>{label}</span>
                      <span className="truncate text-right font-mono" style={{ color: "var(--text-1)" }} title={value}>{value}</span>
                    </div>
                  ))}
                </section>

                {selectedLoaded && (
                  <section className="mt-3 rounded-xl px-3 py-3" style={{ background: "rgba(52,211,153,0.055)", border: "1px solid rgba(52,211,153,0.16)" }}>
                    <div className="flex items-center gap-2 text-[10px] font-semibold" style={{ color: "#6ee7b7" }}><Server size={12} />Active runtime</div>
                    <p className="mt-1.5 text-[10px] leading-4" style={{ color: "var(--text-1)" }}>State {processStatus?.state ?? "Running"}. Reload options lets you change context, GPU offload, batching, templates, and KV settings.</p>
                  </section>
                )}
              </div>
            ) : (
              <div className="flex min-h-0 items-center justify-center px-6 text-center">
                <div>
                  <HardDrive size={22} className="mx-auto" style={{ color: "var(--text-3)" }} />
                  <p className="mt-3 text-xs" style={{ color: "var(--text-2)" }}>Select a model to see its details.</p>
                </div>
              </div>
            )}

            <footer className="border-t px-4 py-3 sm:px-5" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
              {loadDisabledReason && selectedModel && (
                <p className="mb-2 text-[10px] leading-4" role="status" style={{ color: switchingDisabledReason ? "#fcd34d" : "var(--text-2)" }}>{loadDisabledReason}</p>
              )}
              <div className="flex items-center gap-2">
                <Button size="sm" variant="secondary" onClick={onOpenLibrary} icon={<Library size={12} />}>Library</Button>
                <Button
                  size="sm"
                  variant="primary"
                  className="ml-auto"
                  icon={isLoading ? <LoaderCircle size={12} className="animate-spin" /> : selectedLoaded ? <RefreshCw size={12} /> : <ArrowRight size={12} />}
                  disabled={!selectedModel || !!loadDisabledReason}
                  onClick={() => selectedModel && onConfigureLoad(selectedModel)}
                >
                  {selectedModel ? loadActionLabel(selectedModel, loadedModel) : "Select a model"}
                </Button>
              </div>
            </footer>
          </aside>
        </div>
      </div>
    </dialog>
  );
}

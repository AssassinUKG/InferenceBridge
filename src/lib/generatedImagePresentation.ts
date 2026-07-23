export interface GeneratedImageMetadata {
  job_id?: string;
  bundle_id?: string;
  bundle_name?: string;
  quantization?: string;
  profile_id?: string;
  prompt?: string;
  negative_prompt?: string | null;
  seed?: number;
  width?: number;
  height?: number;
  steps?: number;
  cfg_scale?: number;
  sampling_method?: string;
  elapsed_seconds?: number;
  file_size_bytes?: number | null;
  completed_at?: string;
}

export function parseGeneratedImageMetadata(
  value: string | null | undefined,
): GeneratedImageMetadata | null {
  if (!value?.trim()) return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as GeneratedImageMetadata
      : null;
  } catch {
    return null;
  }
}

export function formatImageDuration(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return null;
  if (seconds < 1) return "<1s";
  const rounded = Math.round(seconds);
  const hours = Math.floor(rounded / 3_600);
  const minutes = Math.floor((rounded % 3_600) / 60);
  const remainder = rounded % 60;
  if (hours > 0) return `${hours}h ${minutes}m ${remainder}s`;
  if (minutes > 0) return `${minutes}m ${remainder}s`;
  return `${remainder}s`;
}

export function formatImageFileSize(bytes: number | null | undefined) {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return null;
  if (bytes < 1_024) return `${Math.round(bytes)} B`;
  if (bytes < 1_024 * 1_024) return `${(bytes / 1_024).toFixed(1)} KB`;
  return `${(bytes / (1_024 * 1_024)).toFixed(1)} MB`;
}

export function imageDataUrlByteSize(source: string | null | undefined) {
  if (!source) return null;
  const comma = source.indexOf(",");
  if (comma < 0 || !source.slice(0, comma).includes(";base64")) return null;
  const payload = source.slice(comma + 1);
  const padding = payload.endsWith("==") ? 2 : payload.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor((payload.length * 3) / 4) - padding);
}

export function imageAspectRatio(width: number | null | undefined, height: number | null | undefined) {
  if (!width || !height || width <= 0 || height <= 0) return null;
  const ratio = width / height;
  const known = [
    ["1:1", 1],
    ["16:9", 16 / 9],
    ["9:16", 9 / 16],
    ["4:3", 4 / 3],
    ["3:4", 3 / 4],
    ["3:2", 3 / 2],
    ["2:3", 2 / 3],
  ] as const;
  return known.find(([, candidate]) => Math.abs(ratio - candidate) <= 0.025)?.[0] ?? null;
}

export function imageModelLabel(metadata: GeneratedImageMetadata | null) {
  if (!metadata) return "Qwen-Image";
  let name = metadata.bundle_name?.trim();
  if (!name && metadata.bundle_id?.startsWith("qwen-image-2512")) {
    name = "Qwen-Image 2512";
  }
  name ||= metadata.bundle_id?.trim() || "Qwen-Image";
  name = name.replace(/^Qwen-Image-2512/i, "Qwen-Image 2512");
  const quantization =
    metadata.quantization?.trim().replace(/_K$/i, "") ||
    metadata.bundle_id?.match(/(?:^|-)q(\d+)(?:-|$)/i)?.[0]?.replace(/^-/, "").toUpperCase();
  const nameAlreadyIncludesQuantization =
    !!quantization &&
    name.toLowerCase().replace(/[^a-z0-9]/g, "").includes(quantization.toLowerCase());
  return quantization && !nameAlreadyIncludesQuantization
    ? `${name} · ${quantization}`
    : name;
}

export function formatImageSampler(value: string | null | undefined) {
  if (!value?.trim()) return null;
  const normalized = value.trim().toLowerCase();
  const labels: Record<string, string> = {
    euler: "Euler",
    euler_a: "Euler A",
    heun: "Heun",
    dpm2: "DPM2",
    "dpm++2m": "DPM++ 2M",
  };
  return labels[normalized] ?? value;
}

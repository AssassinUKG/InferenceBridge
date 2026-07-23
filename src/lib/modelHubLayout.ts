export const HUB_LIST_MIN_WIDTH = 320;
export const HUB_DETAIL_MIN_WIDTH = 520;
export const HUB_SEPARATOR_WIDTH = 8;

const DEFAULT_LIST_FRACTION = 0.36;

/**
 * A deliberately small structural type so the recommendation remains useful
 * for API results without depending on download or installation state.
 */
export interface QuantCandidate {
  quant: string;
  filename?: string | null;
}

const BALANCED_QUANT_PREFERENCE = [
  "Q4_K_M",
  "Q4_K",
  "Q5_K_M",
  "Q5_K_S",
  "Q6_K",
  "Q4_K_S",
  "Q8_0",
  "Q4_0",
  "IQ4_XS",
  "Q3_K_M",
  "Q3_K_S",
  "Q2_K",
  "F16",
  "BF16",
] as const;

function usableContainerWidth(containerWidth: number) {
  return Number.isFinite(containerWidth) ? Math.max(0, containerWidth) : 0;
}

function maximumListWidth(containerWidth: number) {
  return Math.max(
    HUB_LIST_MIN_WIDTH,
    usableContainerWidth(containerWidth) - HUB_DETAIL_MIN_WIDTH - HUB_SEPARATOR_WIDTH,
  );
}

/** Return the LM Studio-style initial split, while preserving both panel minima. */
export function defaultHubListWidth(containerWidth: number) {
  const requestedWidth = Math.round(usableContainerWidth(containerWidth) * DEFAULT_LIST_FRACTION);
  return clampHubListWidth(containerWidth, requestedWidth);
}

/** Keep the list within the space left after reserving the detail panel. */
export function clampHubListWidth(containerWidth: number, requestedWidth: number) {
  const fallbackWidth = Math.round(usableContainerWidth(containerWidth) * DEFAULT_LIST_FRACTION);
  const width = Number.isFinite(requestedWidth) ? requestedWidth : fallbackWidth;
  return Math.min(maximumListWidth(containerWidth), Math.max(HUB_LIST_MIN_WIDTH, width));
}

/** Apply a pointer drag delta to the width captured at drag start. */
export function resizeHubListWidth(startWidth: number, deltaX: number, containerWidth: number) {
  const safeStart = Number.isFinite(startWidth) ? startWidth : defaultHubListWidth(containerWidth);
  const safeDelta = Number.isFinite(deltaX) ? deltaX : 0;
  return clampHubListWidth(containerWidth, safeStart + safeDelta);
}

/**
 * Keyboard support for the split separator. Arrow keys resize one step, while
 * Home and End move to the smallest and largest valid list widths.
 */
export function adjustHubListWidthByKeyboard(
  currentWidth: number,
  key: string,
  containerWidth: number,
  step = 24,
) {
  const safeStep = Number.isFinite(step) ? Math.abs(step) : 24;
  const current = clampHubListWidth(containerWidth, currentWidth);

  switch (key) {
    case "ArrowLeft":
      return clampHubListWidth(containerWidth, current - safeStep);
    case "ArrowRight":
      return clampHubListWidth(containerWidth, current + safeStep);
    case "Home":
      return HUB_LIST_MIN_WIDTH;
    case "End":
      return maximumListWidth(containerWidth);
    default:
      return current;
  }
}

function normalizedQuant(value: string) {
  return value.trim().toUpperCase().replace(/-/g, "_");
}

/**
 * Choose a predictable balanced quant from the files a repository actually
 * offers. Installation state is intentionally absent: installing a file must
 * not silently change which quant is labelled as recommended.
 */
export function chooseRecommendedQuant<T extends QuantCandidate>(quants: readonly T[]): T | null {
  for (const preferred of BALANCED_QUANT_PREFERENCE) {
    const match = quants.find((candidate) => normalizedQuant(candidate.quant) === preferred);
    if (match) return match;
  }

  return quants[0] ?? null;
}

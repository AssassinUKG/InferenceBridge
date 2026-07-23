import type { LiveStreamDelta, LiveStreamEvent, LiveStreamSnapshot } from "./types";

export const LIVE_STREAM_EVENT_LIMIT = 500;
export const LIVE_STREAM_HISTORY_LIMIT = 30;

function eventKey(event: LiveStreamEvent) {
  return `${event.timestamp}\u0000${event.kind}\u0000${event.text}`;
}

function richerText(current: string, incoming: string) {
  return incoming.length >= current.length ? incoming : current;
}

function isTerminalStatus(status: string) {
  return !isLiveStreamRunningStatus(status);
}

function reconciledStatus(current: string, incoming: string) {
  // A slow status poll must not turn a request that already completed locally
  // back into a running request. A terminal incoming status is authoritative.
  if (isTerminalStatus(current) && !isTerminalStatus(incoming)) return current;
  return incoming;
}

export function isLiveStreamRunningStatus(status: string) {
  const normalized = status.trim().toLowerCase();
  return normalized === "running" || normalized.startsWith("streaming");
}

export function liveStreamHasDelta(stream: LiveStreamSnapshot, delta: LiveStreamDelta) {
  const key = eventKey(delta);
  return stream.events.some((event) => eventKey(event) === key);
}

export function appendLiveStreamDelta(
  stream: LiveStreamSnapshot,
  delta: LiveStreamDelta,
): LiveStreamSnapshot {
  if (liveStreamHasDelta(stream, delta)) return stream;

  return {
    ...stream,
    raw_output:
      delta.kind === "raw" || delta.kind === "error"
        ? stream.raw_output + delta.text
        : stream.raw_output,
    visible_output:
      delta.kind === "content"
        ? stream.visible_output + delta.text
        : stream.visible_output,
    reasoning_output:
      delta.kind === "reasoning"
        ? stream.reasoning_output + delta.text
        : stream.reasoning_output,
    events: [...stream.events, delta].slice(-LIVE_STREAM_EVENT_LIMIT),
  };
}

export function reconcileLiveStream(
  current: LiveStreamSnapshot,
  incoming: LiveStreamSnapshot,
): LiveStreamSnapshot {
  if (current.request_id !== incoming.request_id) return incoming;

  const events = [...current.events];
  const seen = new Set(events.map(eventKey));
  for (const event of incoming.events) {
    const key = eventKey(event);
    if (!seen.has(key)) {
      seen.add(key);
      events.push(event);
    }
  }
  events.sort((a, b) => a.timestamp.localeCompare(b.timestamp));

  return {
    ...incoming,
    status: reconciledStatus(current.status, incoming.status),
    raw_output: richerText(current.raw_output, incoming.raw_output),
    visible_output: richerText(current.visible_output, incoming.visible_output),
    reasoning_output: richerText(current.reasoning_output, incoming.reasoning_output),
    events: events.slice(-LIVE_STREAM_EVENT_LIMIT),
  };
}

export function upsertLiveStream(
  current: LiveStreamSnapshot[],
  incoming: LiveStreamSnapshot,
): LiveStreamSnapshot[] {
  const index = current.findIndex((stream) => stream.request_id === incoming.request_id);
  if (index < 0) {
    return [...current, incoming].slice(-LIVE_STREAM_HISTORY_LIMIT);
  }

  const next = [...current];
  next[index] = reconcileLiveStream(current[index], incoming);
  return next;
}

export function reconcileLiveStreams(
  current: LiveStreamSnapshot[],
  incoming: LiveStreamSnapshot[],
) {
  let next = current;
  for (const stream of incoming) {
    next = upsertLiveStream(next, stream);
  }
  return next.slice(-LIVE_STREAM_HISTORY_LIMIT);
}

export function applyLiveStreamDelta(
  current: LiveStreamSnapshot[],
  delta: LiveStreamDelta,
) {
  return current.map((stream) =>
    stream.request_id === delta.request_id
      ? appendLiveStreamDelta(stream, delta)
      : stream,
  );
}

export function activeLiveStream(streams: LiveStreamSnapshot[]) {
  let active: LiveStreamSnapshot | null = null;
  let activeTime = Number.NEGATIVE_INFINITY;

  for (const stream of streams) {
    if (!isLiveStreamRunningStatus(stream.status)) continue;
    const parsed = Date.parse(stream.started_at);
    const started = Number.isNaN(parsed) ? 0 : parsed;
    if (!active || started >= activeTime) {
      active = stream;
      activeTime = started;
    }
  }

  return active;
}

export function pinLiveStream(
  streams: LiveStreamSnapshot[],
  stream: LiveStreamSnapshot | null,
) {
  if (!stream) return streams;
  const index = streams.findIndex((entry) => entry.request_id === stream.request_id);
  if (index <= 0) return streams;
  return [streams[index], ...streams.slice(0, index), ...streams.slice(index + 1)];
}

export function tailLiveStream(
  streams: LiveStreamSnapshot[],
  stream: LiveStreamSnapshot | null,
) {
  if (!stream) return streams;
  const index = streams.findIndex((entry) => entry.request_id === stream.request_id);
  if (index < 0 || index === streams.length - 1) return streams;
  return [...streams.slice(0, index), ...streams.slice(index + 1), streams[index]];
}

export type LiveStreamLogLevel = "INFO" | "LIVE" | "ERROR";

export type LiveStreamTerminalKind = "INPUT" | "OUTPUT" | "THINK" | "TOOL" | "ERROR";

export interface LiveStreamTerminalRow {
  timestamp: string;
  kind: LiveStreamTerminalKind;
  source: string;
  direction: "->" | "<-";
  message: string;
}

export function liveStreamLogLevel(status: string): LiveStreamLogLevel {
  if (isLiveStreamRunningStatus(status)) return "LIVE";
  return status.trim().toLowerCase() === "error" ? "ERROR" : "INFO";
}

export function liveStreamLogSource(source: string) {
  const normalized = source.trim();
  return normalized ? normalized.toUpperCase() : "UNKNOWN";
}

export function liveStreamInputText(stream: LiveStreamSnapshot) {
  return stream.events
    .filter((event) => event.kind === "input")
    .map((event) => event.text.trim())
    .filter(Boolean)
    .join("\n");
}

function terminalLines(text: string) {
  return text
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line) => line.trim().length > 0);
}

export function liveStreamTerminalRows(stream: LiveStreamSnapshot): LiveStreamTerminalRow[] {
  const rows: Array<LiveStreamTerminalRow & { order: number }> = [];
  let order = 0;
  const pushLines = (
    timestamp: string,
    kind: LiveStreamTerminalKind,
    source: string,
    direction: "->" | "<-",
    text: string,
  ) => {
    for (const message of terminalLines(text)) {
      rows.push({ timestamp, kind, source, direction, message, order });
      order += 1;
    }
  };

  for (const event of stream.events) {
    if (event.kind === "input") {
      pushLines(event.timestamp, "INPUT", "USER", "->", event.text);
    } else if (event.kind === "tool_call") {
      pushLines(event.timestamp, "TOOL", "MODEL", "->", event.text);
    } else if (event.kind === "error") {
      pushLines(event.timestamp, "ERROR", "RUNTIME", "<-", event.text);
    }
  }

  const reasoning = stream.reasoning_output.trim();
  if (reasoning) {
    const timestamp = stream.events.find((event) => event.kind === "reasoning")?.timestamp
      ?? stream.started_at;
    pushLines(timestamp, "THINK", "MODEL", "<-", reasoning);
  }

  const bufferedOutput = stream.events
    .filter((event) => event.kind === "content_buffered")
    .map((event) => event.text)
    .join("");
  const visibleOutput = stream.visible_output.trim() || bufferedOutput.trim();
  if (visibleOutput) {
    const timestamp = stream.events.find((event) =>
      event.kind === "content" || event.kind === "content_buffered"
    )?.timestamp ?? stream.started_at;
    pushLines(timestamp, "OUTPUT", "MODEL", "<-", visibleOutput);
  }

  if (stream.status.trim().toLowerCase() === "error" && !rows.some((row) => row.kind === "ERROR")) {
    pushLines(
      stream.events.at(-1)?.timestamp ?? stream.started_at,
      "ERROR",
      "RUNTIME",
      "<-",
      stream.raw_output || "Generation failed without an error message.",
    );
  }

  return rows
    .sort((left, right) => {
      const timestampOrder = left.timestamp.localeCompare(right.timestamp);
      return timestampOrder === 0 ? left.order - right.order : timestampOrder;
    })
    .map(({ order: _order, ...row }) => row);
}

export function formatLiveStreamTranscript(
  input: string,
  output: string,
  outputLabel: "OUTPUT" | "RAW OUTPUT" | "REASONING" = "OUTPUT",
) {
  const blocks: string[] = [];
  if (input.trim()) blocks.push(`[INPUT]\n${input.trim()}`);
  if (output.trim()) blocks.push(`[${outputLabel}]\n${output.trim()}`);
  return blocks.join("\n\n");
}

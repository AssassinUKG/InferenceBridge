import assert from "node:assert/strict";
import test from "node:test";

import {
  LIVE_STREAM_EVENT_LIMIT,
  activeLiveStream,
  appendLiveStreamDelta,
  formatLiveStreamTranscript,
  liveStreamInputText,
  liveStreamLogLevel,
  liveStreamLogSource,
  liveStreamTerminalRows,
  pinLiveStream,
  reconcileLiveStream,
  reconcileLiveStreams,
  tailLiveStream,
} from "../src/lib/liveStreams.ts";

const snapshot = (requestId, extra = {}) => ({
  request_id: requestId,
  source: "api",
  model: "test.gguf",
  started_at: "2026-07-16T10:00:00.000Z",
  status: "running",
  raw_output: "",
  visible_output: "",
  reasoning_output: "",
  events: [],
  ...extra,
});

const delta = (requestId, kind, text, timestamp = "2026-07-16T10:00:01.000Z") => ({
  request_id: requestId,
  timestamp,
  kind,
  text,
});

test("live deltas update the matching output once", () => {
  const initial = snapshot("request-1");
  const raw = delta("request-1", "raw", "raw token");
  const withRaw = appendLiveStreamDelta(initial, raw);
  const duplicate = appendLiveStreamDelta(withRaw, raw);
  const withContent = appendLiveStreamDelta(
    duplicate,
    delta("request-1", "content", "hello", "2026-07-16T10:00:02.000Z"),
  );
  const withReasoning = appendLiveStreamDelta(
    withContent,
    delta("request-1", "reasoning", "think", "2026-07-16T10:00:03.000Z"),
  );

  assert.equal(withReasoning.raw_output, "raw token");
  assert.equal(withReasoning.visible_output, "hello");
  assert.equal(withReasoning.reasoning_output, "think");
  assert.equal(withReasoning.events.length, 3);
});

test("local live deltas are capped to the backend history limit", () => {
  let stream = snapshot("request-cap");
  for (let index = 0; index < LIVE_STREAM_EVENT_LIMIT + 25; index += 1) {
    stream = appendLiveStreamDelta(
      stream,
      delta(
        "request-cap",
        "raw",
        String(index),
        `2026-07-16T10:${String(Math.floor(index / 60)).padStart(2, "0")}:${String(index % 60).padStart(2, "0")}.000Z`,
      ),
    );
  }

  assert.equal(stream.events.length, LIVE_STREAM_EVENT_LIMIT);
  assert.equal(stream.events[0].text, "25");
});

test("a stale poll cannot roll back event-driven output or terminal status", () => {
  const local = snapshot("request-2", {
    status: "completed",
    raw_output: "complete local output",
    visible_output: "complete local output",
    events: [delta("request-2", "raw", "complete local output")],
  });
  const stalePoll = snapshot("request-2", {
    status: "running",
    raw_output: "complete",
    visible_output: "complete",
    events: [],
  });

  const merged = reconcileLiveStream(local, stalePoll);
  assert.equal(merged.status, "completed");
  assert.equal(merged.raw_output, "complete local output");
  assert.equal(merged.visible_output, "complete local output");
  assert.equal(merged.events.length, 1);
});

test("a terminal poll can complete a locally running stream without dropping output", () => {
  const local = snapshot("request-3", {
    raw_output: "live token",
    events: [delta("request-3", "raw", "live token")],
  });
  const completedPoll = snapshot("request-3", { status: "completed" });

  const merged = reconcileLiveStreams([local], [completedPoll]);
  assert.equal(merged[0].status, "completed");
  assert.equal(merged[0].raw_output, "live token");
  assert.equal(merged[0].events.length, 1);
});

test("the newest running request is selected and pinned ahead of history", () => {
  const completed = snapshot("completed", {
    status: "completed",
    started_at: "2026-07-16T10:02:00.000Z",
  });
  const olderRunning = snapshot("older-running", {
    started_at: "2026-07-16T10:00:00.000Z",
  });
  const newestRunning = snapshot("newest-running", {
    status: "streaming 4 token(s)",
    started_at: "2026-07-16T10:01:00.000Z",
  });
  const streams = [olderRunning, completed, newestRunning];

  const active = activeLiveStream(streams);
  assert.equal(active?.request_id, "newest-running");
  assert.deepEqual(
    pinLiveStream(streams, active).map((stream) => stream.request_id),
    ["newest-running", "older-running", "completed"],
  );
});

test("terminal logs keep the live request at the chronological tail", () => {
  const active = snapshot("active");
  const streams = [active, snapshot("done-1", { status: "completed" }), snapshot("done-2", { status: "completed" })];

  assert.deepEqual(
    tailLiveStream(streams, active).map((stream) => stream.request_id),
    ["done-1", "done-2", "active"],
  );
  assert.equal(tailLiveStream(streams, null), streams);
});

test("terminal log metadata normalizes levels and source labels", () => {
  assert.equal(liveStreamLogLevel("running"), "LIVE");
  assert.equal(liveStreamLogLevel("streaming 4 token(s)"), "LIVE");
  assert.equal(liveStreamLogLevel("error"), "ERROR");
  assert.equal(liveStreamLogLevel("completed"), "INFO");
  assert.equal(liveStreamLogSource(" gui "), "GUI");
  assert.equal(liveStreamLogSource(""), "UNKNOWN");
});

test("terminal transcript shows user input before model output", () => {
  const stream = snapshot("request-with-input", {
    events: [delta("request-with-input", "input", "what I put")],
  });

  assert.equal(liveStreamInputText(stream), "what I put");
  assert.equal(
    formatLiveStreamTranscript(liveStreamInputText(stream), "the reply"),
    "[INPUT]\nwhat I put\n\n[OUTPUT]\nthe reply",
  );
});

test("structured terminal rows keep input before concise model output", () => {
  const stream = snapshot("request-terminal", {
    status: "completed",
    raw_output: "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}",
    visible_output: "hello\nthere",
    events: [
      delta("request-terminal", "input", "what I put", "2026-07-16T10:00:01.000Z"),
      delta("request-terminal", "raw", "a large raw payload", "2026-07-16T10:00:02.000Z"),
      delta("request-terminal", "content", "hello", "2026-07-16T10:00:03.000Z"),
    ],
  });

  assert.deepEqual(
    liveStreamTerminalRows(stream).map((row) => [row.kind, row.source, row.direction, row.message]),
    [
      ["INPUT", "USER", "->", "what I put"],
      ["OUTPUT", "MODEL", "<-", "hello"],
      ["OUTPUT", "MODEL", "<-", "there"],
    ],
  );
});

test("structured terminal rows classify reasoning, tools, and errors without dumping raw chunks", () => {
  const stream = snapshot("request-events", {
    status: "error",
    reasoning_output: "checking the request",
    raw_output: "raw SSE should not become an ordinary row",
    events: [
      delta("request-events", "reasoning", "checking", "2026-07-16T10:00:01.000Z"),
      delta("request-events", "tool_call", '{"name":"whoami"}', "2026-07-16T10:00:02.000Z"),
      delta("request-events", "error", "command failed", "2026-07-16T10:00:03.000Z"),
      delta("request-events", "raw", "ignored", "2026-07-16T10:00:04.000Z"),
    ],
  });

  const rows = liveStreamTerminalRows(stream);
  assert.deepEqual(rows.map((row) => row.kind), ["THINK", "TOOL", "ERROR"]);
  assert.equal(rows.some((row) => row.message.includes("raw SSE")), false);
});

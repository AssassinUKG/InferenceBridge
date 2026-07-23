import assert from "node:assert/strict";
import test from "node:test";

import {
  createOptimisticUserMessage,
  latestAssistantOutputTokens,
  reconcileLoadedSessionMessages,
} from "../src/lib/chatPresentation.ts";

test("optimistic user turn is renderable before inference completes", () => {
  const existing = [{ id: 1, role: "assistant", content: "Ready" }];
  const pending = createOptimisticUserMessage(
    -1,
    "show immediately",
    "data:image/png;base64,QUFBQQ==",
    "2026-07-17T12:00:00.000Z",
  );
  const visible = [...existing, pending];

  assert.equal(visible.at(-1), pending);
  assert.equal(pending.role, "user");
  assert.equal(pending.content, "show immediately");
  assert.equal(pending.image_base64, "data:image/png;base64,QUFBQQ==");
  assert.equal(pending.created_at, "2026-07-17T12:00:00.000Z");
  assert.deepEqual(pending.tool_calls, []);
});

test("chat header uses the latest stored assistant output count", () => {
  const messages = [
    { id: 1, role: "assistant", token_count: 17, tokens_predicted: 17 },
    { id: 2, role: "user", token_count: 0, tokens_predicted: null },
    { id: 3, role: "assistant", token_count: 160, tokens_predicted: 162 },
    createOptimisticUserMessage(-1, "next prompt", null),
  ];

  assert.equal(latestAssistantOutputTokens(messages), 162);
  assert.equal(latestAssistantOutputTokens([{ id: 4, role: "user" }]), null);
});

test("a stale session load cannot remove the optimistic prompt during generation", () => {
  const earlier = {
    ...createOptimisticUserMessage(10, "earlier", null, "2026-07-17T12:00:00.000Z"),
    role: "assistant",
  };
  const optimistic = createOptimisticUserMessage(
    -1,
    "show before the reply",
    null,
    "2026-07-17T12:01:00.000Z",
  );

  const reconciled = reconcileLoadedSessionMessages(
    [earlier, optimistic],
    [earlier],
    true,
  );

  assert.equal(reconciled.at(-1), optimistic);
  assert.equal(reconciled.at(-1)?.content, "show before the reply");
});

test("session reconciliation replaces an optimistic prompt with its persisted copy", () => {
  const optimistic = createOptimisticUserMessage(
    -1,
    "persisted promptly",
    null,
    "2026-07-17T12:01:00.000Z",
  );
  const persisted = {
    ...optimistic,
    id: 42,
    created_at: "2026-07-17T12:01:01.000Z",
  };

  const reconciled = reconcileLoadedSessionMessages(
    [optimistic],
    [persisted],
    true,
  );

  assert.deepEqual(reconciled, [persisted]);
});

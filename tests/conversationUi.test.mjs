import assert from "node:assert/strict";
import test from "node:test";

import {
  composerPrimaryAction,
  conversationMarkdown,
  isNearScrollBottom,
  safeConversationFilename,
} from "../src/lib/conversationUi.ts";

test("uses the primary composer action for image generation without a chat model", () => {
  assert.equal(composerPrimaryAction(true, true), "send_message");
  assert.equal(composerPrimaryAction(true, false), "send_message");
  assert.equal(composerPrimaryAction(false, true), "generate_image");
  assert.equal(composerPrimaryAction(false, false), "unavailable");
});

test("only follows streaming output while the reader is near the bottom", () => {
  assert.equal(isNearScrollBottom(2_000, 1_300, 600), false);
  assert.equal(isNearScrollBottom(2_000, 1_304, 600), true);
  assert.equal(isNearScrollBottom(2_000, 1_400, 600), true);
});

test("sanitizes exported conversation filenames", () => {
  assert.equal(safeConversationFilename('Plan: tools/loops?*'), "Plan tools loops.md");
  assert.equal(safeConversationFilename("   "), "InferenceBridge chat.md");
});

test("exports visible content and tool selections without hidden reasoning", () => {
  const markdown = conversationMarkdown(
    { id: "one", name: "Tool test", model_id: null, pinned: false, created_at: "", updated_at: "" },
    [{
      id: 1,
      role: "assistant",
      content: "<think>private</think>raw",
      display_content: "Visible answer",
      reasoning_content: "private",
      image_base64: null,
      token_count: 2,
      created_at: "",
      tool_calls: [{ id: 1, call_id: "call_1", name: "clock", arguments: '{"zone":"UTC"}', result: null }],
    }],
  );

  assert.match(markdown, /Visible answer/);
  assert.match(markdown, /clock/);
  assert.doesNotMatch(markdown, /private/);
});

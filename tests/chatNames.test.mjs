import assert from "node:assert/strict";
import test from "node:test";

import { nextAutomaticChatName } from "../src/lib/chatNames.ts";

const names = (...values) => values.map((name) => ({ name }));

test("starts automatic chat numbering at one", () => {
  assert.equal(nextAutomaticChatName([]), "Chat 1");
});

test("uses the automatic-chat count instead of the highest historical suffix", () => {
  assert.equal(nextAutomaticChatName(names("Chat 9", "Chat 1")), "Chat 3");
});

test("continues a contiguous sequence after deletion compacts the remaining chats", () => {
  const beforeDeletion = names("Chat 1", "Chat 2", "Chat 3");
  const afterDeletion = beforeDeletion
    .filter((session) => session.name !== "Chat 2")
    .map((session, index) => ({ ...session, name: `Chat ${index + 1}` }));

  assert.deepEqual(afterDeletion.map((session) => session.name), ["Chat 1", "Chat 2"]);
  assert.equal(nextAutomaticChatName(afterDeletion), "Chat 3");
});

test("does not count or alter user-provided titles", () => {
  const sessions = names("Release notes", "Chat 2", "Customer follow-up", "Chat 1");

  assert.equal(
    nextAutomaticChatName(sessions),
    "Chat 3",
  );
  assert.deepEqual(
    sessions.map((session) => session.name),
    ["Release notes", "Chat 2", "Customer follow-up", "Chat 1"],
  );
});

test("only exact automatic titles participate in the count", () => {
  assert.equal(
    nextAutomaticChatName(names("Chatty 99", "chat 8", "Chat 2", "Chat 1", null)),
    "Chat 3",
  );
});

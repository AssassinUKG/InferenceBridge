import type { MessageInfo } from "./types";

export function createOptimisticUserMessage(
  id: number,
  content: string,
  imageBase64: string | null | undefined,
  createdAt = new Date().toISOString(),
): MessageInfo {
  return {
    id,
    role: "user",
    content,
    display_content: null,
    reasoning_content: null,
    image_base64: imageBase64 ?? null,
    token_count: 0,
    tokens_evaluated: null,
    tokens_predicted: null,
    created_at: createdAt,
    tool_calls: [],
  };
}

export function latestAssistantOutputTokens(messages: readonly MessageInfo[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "assistant") continue;
    return message.tokens_predicted ?? message.token_count ?? null;
  }
  return null;
}

function timestampsAreClose(left: string, right: string) {
  const leftMs = Date.parse(left);
  const rightMs = Date.parse(right);
  if (Number.isNaN(leftMs) || Number.isNaN(rightMs)) return false;
  return Math.abs(leftMs - rightMs) <= 60_000;
}

function persistedCopyOfOptimisticMessage(
  persisted: MessageInfo,
  optimistic: MessageInfo,
) {
  return (
    persisted.id >= 0 &&
    persisted.role === "user" &&
    optimistic.role === "user" &&
    persisted.content === optimistic.content &&
    (persisted.image_base64 ?? null) === (optimistic.image_base64 ?? null) &&
    timestampsAreClose(persisted.created_at, optimistic.created_at)
  );
}

export function reconcileLoadedSessionMessages(
  current: readonly MessageInfo[],
  loaded: readonly MessageInfo[],
  generationPending: boolean,
) {
  if (!generationPending) return [...loaded];

  const merged = [...loaded];
  const loadedIds = new Set(loaded.map((message) => message.id));

  for (const message of current) {
    if (message.id >= 0 && !loadedIds.has(message.id)) merged.push(message);
  }

  const optimisticMessages = current.filter((message) => message.id < 0);
  for (const optimistic of optimisticMessages) {
    const alreadyPersisted = merged.some((message) =>
      persistedCopyOfOptimisticMessage(message, optimistic)
    );
    if (!alreadyPersisted) merged.push(optimistic);
  }

  return merged.sort((left, right) => {
    const leftMs = Date.parse(left.created_at);
    const rightMs = Date.parse(right.created_at);
    if (Number.isNaN(leftMs) || Number.isNaN(rightMs)) return left.id - right.id;
    return leftMs - rightMs;
  });
}

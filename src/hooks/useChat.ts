import { useState, useCallback, useEffect } from "react";
import { flushSync } from "react-dom";
import { listen } from "@tauri-apps/api/event";
import type { MessageInfo } from "../lib/types";
import * as api from "../lib/tauri";
import type { SamplingParams } from "../lib/tauri";

interface ChatState {
  messages: MessageInfo[];
  isStreaming: boolean;
  streamingText: string;
  tokensPerSecond: number | null;
  error: string | null;
}

export function useChat(sessionId: string | null) {
  const [state, setState] = useState<ChatState>({
    messages: [],
    isStreaming: false,
    streamingText: "",
    tokensPerSecond: null,
    error: null,
  });

  useEffect(() => {
    if (!sessionId) {
      setState((s) => ({ ...s, messages: [] }));
      return;
    }
    api.getSessionMessages(sessionId).then(
      (msgs) => setState((s) => ({ ...s, messages: msgs, error: null })),
      (err) => setState((s) => ({ ...s, error: String(err) }))
    );
  }, [sessionId]);

  // Note: stream-error is handled per-message inside sendMessage to avoid
  // duplicate listeners and stale-closure bugs. No persistent global listener needed.

  const sendMessage = useCallback(
    async (
      content: string,
      sampling?: SamplingParams,
      imageBase64?: string | null,
      showThinking?: boolean | null
    ) => {
      if (!sessionId || state.isStreaming) return;
      setState((s) => ({
        ...s,
        isStreaming: true,
        streamingText: "",
        tokensPerSecond: null,
        error: null,
      }));

      const cleanups: (() => void)[] = [];

      try {
        // Set up event listeners before invoking the command so we do not miss early tokens.
        const tokenUnsub = await listen<string>("stream-token", (e) => {
          // flushSync forces React to render immediately after each token rather
          // than batching multiple tokens into a single render (React 18 behaviour).
          // This gives true per-token streaming instead of chunk-bursts.
          flushSync(() => {
            setState((s) => ({
              ...s,
              streamingText: s.streamingText + e.payload,
            }));
          });
        });
        cleanups.push(tokenUnsub);

        const doneUnsub = await listen<number>("stream-done", (e) => {
          setState((s) => ({
            ...s,
            tokensPerSecond: e.payload,
          }));
        });
        cleanups.push(doneUnsub);

        const errorUnsub = await listen<string>("stream-error", (e) => {
          setState((s) => ({
            ...s,
            isStreaming: false,
            error: e.payload,
            streamingText: "",
          }));
        });
        cleanups.push(errorUnsub);

        await api.sendMessage(sessionId, content, sampling, imageBase64, showThinking);
        // Command completed - refresh the full message list.
        const msgs = await api.getSessionMessages(sessionId);
        setState((s) => ({
          ...s,
          messages: msgs,
          streamingText: "",
          isStreaming: false,
        }));
      } catch (err) {
        setState((s) => ({
          ...s,
          isStreaming: false,
          error: String(err),
          streamingText: "",
        }));
      } finally {
        cleanups.forEach((u) => u());
      }
    },
    [sessionId, state.isStreaming]
  );

  const stopGeneration = useCallback(async () => {
    await api.stopGeneration();
    setState((s) => ({ ...s, isStreaming: false }));
  }, []);

  return { ...state, sendMessage, stopGeneration };
}

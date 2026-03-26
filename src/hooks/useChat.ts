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
  streamingReasoning: string;
  tokensPerSecond: number | null;
  error: string | null;
}

export function useChat(sessionId: string | null) {
  const [state, setState] = useState<ChatState>({
    messages: [],
    isStreaming: false,
    streamingText: "",
    streamingReasoning: "",
    tokensPerSecond: null,
    error: null,
  });

  useEffect(() => {
    if (!sessionId) {
      setState((current) => ({ ...current, messages: [] }));
      return;
    }

    api.getSessionMessages(sessionId).then(
      (messages) => setState((current) => ({ ...current, messages, error: null })),
      (error) => setState((current) => ({ ...current, error: String(error) }))
    );
  }, [sessionId]);

  const sendMessage = useCallback(
    async (
      content: string,
      sampling?: SamplingParams,
      imageBase64?: string | null,
      showThinking?: boolean | null
    ) => {
      if (!sessionId || state.isStreaming) return;

      setState((current) => ({
        ...current,
        isStreaming: true,
        streamingText: "",
        streamingReasoning: "",
        tokensPerSecond: null,
        error: null,
      }));

      const cleanups: Array<() => void> = [];

      try {
        const tokenUnsub = await listen<string>("stream-token", (event) => {
          flushSync(() => {
            setState((current) => ({
              ...current,
              streamingText: current.streamingText + event.payload,
            }));
          });
        });
        cleanups.push(tokenUnsub);

        const thinkingUnsub = await listen<string>("stream-thinking", (event) => {
          flushSync(() => {
            setState((current) => ({
              ...current,
              streamingReasoning: current.streamingReasoning + event.payload,
            }));
          });
        });
        cleanups.push(thinkingUnsub);

        const doneUnsub = await listen<number>("stream-done", (event) => {
          setState((current) => ({
            ...current,
            tokensPerSecond: event.payload,
          }));
        });
        cleanups.push(doneUnsub);

        const errorUnsub = await listen<string>("stream-error", (event) => {
          setState((current) => ({
            ...current,
            isStreaming: false,
            error: event.payload,
            streamingText: "",
            streamingReasoning: "",
          }));
        });
        cleanups.push(errorUnsub);

        await api.sendMessage(sessionId, content, sampling, imageBase64, showThinking);
        const messages = await api.getSessionMessages(sessionId);
        setState((current) => ({
          ...current,
          messages,
          streamingText: "",
          streamingReasoning: "",
          isStreaming: false,
        }));
      } catch (error) {
        setState((current) => ({
          ...current,
          isStreaming: false,
          error: String(error),
          streamingText: "",
          streamingReasoning: "",
        }));
      } finally {
        cleanups.forEach((cleanup) => cleanup());
      }
    },
    [sessionId, state.isStreaming]
  );

  const stopGeneration = useCallback(async () => {
    await api.stopGeneration();
    setState((current) => ({
      ...current,
      isStreaming: false,
      streamingText: "",
      streamingReasoning: "",
    }));
  }, []);

  return { ...state, sendMessage, stopGeneration };
}

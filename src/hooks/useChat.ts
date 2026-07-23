import { useState, useCallback, useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import type { MessageInfo } from "../lib/types";
import * as api from "../lib/tauri";
import type { SamplingParams } from "../lib/tauri";
import {
  createOptimisticUserMessage,
  reconcileLoadedSessionMessages,
} from "../lib/chatPresentation";

interface ChatState {
  messages: MessageInfo[];
  isStreaming: boolean;
  streamingText: string;
  streamingReasoning: string;
  tokensPerSecond: number | null;
  error: string | null;
}

export function useChat(sessionId: string | null) {
  const pendingTextRef = useRef("");
  const pendingReasoningRef = useRef("");
  const flushFrameRef = useRef<number | null>(null);
  const optimisticMessageIdRef = useRef(-1);
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

    let cancelled = false;

    api.getSessionMessages(sessionId).then(
      (messages) => {
        if (cancelled) return;
        setState((current) => ({
          ...current,
          messages: reconcileLoadedSessionMessages(
            current.messages,
            messages,
            current.isStreaming,
          ),
          error: null,
        }));
      },
      (error) => {
        if (!cancelled) setState((current) => ({ ...current, error: String(error) }));
      }
    );

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return undefined;
    let mounted = true;
    let stopListening: (() => void) | undefined;
    void listen<string>("session-messages-changed", (event) => {
      if (!mounted || event.payload !== sessionId) return;
      void api.getSessionMessages(sessionId).then(
        (messages) => {
          if (mounted) {
            setState((current) => ({ ...current, messages, error: null }));
          }
        },
        (error) => {
          if (mounted) {
            setState((current) => ({ ...current, error: String(error) }));
          }
        },
      );
    }).then((unlisten) => {
      if (mounted) stopListening = unlisten;
      else unlisten();
    });
    return () => {
      mounted = false;
      stopListening?.();
    };
  }, [sessionId]);

  const flushStreamingBuffers = useCallback(() => {
    flushFrameRef.current = null;
    const text = pendingTextRef.current;
    const reasoning = pendingReasoningRef.current;
    if (!text && !reasoning) return;

    pendingTextRef.current = "";
    pendingReasoningRef.current = "";
    setState((current) => ({
      ...current,
      streamingText: current.streamingText + text,
      streamingReasoning: current.streamingReasoning + reasoning,
    }));
  }, []);

  const scheduleStreamingFlush = useCallback(() => {
    if (flushFrameRef.current !== null) return;
    flushFrameRef.current = requestAnimationFrame(flushStreamingBuffers);
  }, [flushStreamingBuffers]);

  const cancelStreamingFlush = useCallback(() => {
    if (flushFrameRef.current !== null) {
      cancelAnimationFrame(flushFrameRef.current);
      flushFrameRef.current = null;
    }
    pendingTextRef.current = "";
    pendingReasoningRef.current = "";
  }, []);

  useEffect(() => cancelStreamingFlush, [cancelStreamingFlush]);

  const sendMessage = useCallback(
    async (
      content: string,
      sampling?: SamplingParams,
      imageBase64?: string | null,
      showThinking?: boolean | null
    ) => {
      if (!sessionId || state.isStreaming) return;

      const optimisticMessageId = optimisticMessageIdRef.current--;
      const optimisticMessage = createOptimisticUserMessage(
        optimisticMessageId,
        content,
        imageBase64,
      );

      setState((current) => ({
        ...current,
        messages: [...current.messages, optimisticMessage],
        isStreaming: true,
        streamingText: "",
        streamingReasoning: "",
        tokensPerSecond: null,
        error: null,
      }));

      const cleanups: Array<() => void> = [];

      try {
        const tokenUnsub = await listen<string>("stream-token", (event) => {
          pendingTextRef.current += event.payload;
          scheduleStreamingFlush();
        });
        cleanups.push(tokenUnsub);

        const thinkingUnsub = await listen<string>("stream-thinking", (event) => {
          pendingReasoningRef.current += event.payload;
          scheduleStreamingFlush();
        });
        cleanups.push(thinkingUnsub);

        const doneUnsub = await listen<number>("stream-done", (event) => {
          flushStreamingBuffers();
          setState((current) => ({
            ...current,
            tokensPerSecond: event.payload,
          }));
        });
        cleanups.push(doneUnsub);

        const errorUnsub = await listen<string>("stream-error", (event) => {
          cancelStreamingFlush();
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
        flushStreamingBuffers();
        const messages = await api.getSessionMessages(sessionId);
        setState((current) => ({
          ...current,
          messages,
          streamingText: "",
          streamingReasoning: "",
          isStreaming: false,
        }));
      } catch (error) {
        cancelStreamingFlush();
        let persistedMessages: MessageInfo[] | null = null;
        try {
          persistedMessages = await api.getSessionMessages(sessionId);
        } catch {
          // Keep the optimistic prompt visible when reconciliation itself fails.
        }
        setState((current) => ({
          ...current,
          messages: persistedMessages ?? current.messages,
          isStreaming: false,
          error: String(error),
          streamingText: "",
          streamingReasoning: "",
        }));
      } finally {
        cleanups.forEach((cleanup) => cleanup());
      }
    },
    [
      sessionId,
      state.isStreaming,
      scheduleStreamingFlush,
      flushStreamingBuffers,
      cancelStreamingFlush,
    ]
  );

  const stopGeneration = useCallback(async () => {
    await api.stopGeneration();
    cancelStreamingFlush();
    setState((current) => ({
      ...current,
      isStreaming: false,
      streamingText: "",
      streamingReasoning: "",
    }));
  }, [cancelStreamingFlush]);

  return { ...state, sendMessage, stopGeneration };
}

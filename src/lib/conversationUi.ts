import type { MessageInfo, SessionInfo } from "./types";

export const STREAM_FOLLOW_THRESHOLD_PX = 96;

export type ComposerPrimaryAction = "send_message" | "generate_image" | "unavailable";

export function composerPrimaryAction(
  hasModel: boolean,
  imageGenerationReady: boolean,
): ComposerPrimaryAction {
  if (hasModel) return "send_message";
  if (imageGenerationReady) return "generate_image";
  return "unavailable";
}

export function isNearScrollBottom(
  scrollHeight: number,
  scrollTop: number,
  clientHeight: number,
  threshold = STREAM_FOLLOW_THRESHOLD_PX,
) {
  return scrollHeight - scrollTop - clientHeight <= Math.max(0, threshold);
}

export function safeConversationFilename(name: string | null | undefined) {
  const base = (name || "InferenceBridge chat")
    .normalize("NFKC")
    .replace(/[<>:"/\\|?*\u0000-\u001F]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[. ]+$/g, "")
    .slice(0, 80);
  return `${base || "InferenceBridge chat"}.md`;
}

export function conversationMarkdown(session: SessionInfo, messages: MessageInfo[]) {
  const title = session.name?.trim() || "InferenceBridge chat";
  const body = messages.map((message) => {
    const role = message.role === "assistant" ? "InferenceBridge" :
      message.role === "user" ? "User" : "System";
    const visible = message.display_content ?? message.content ?? "";
    const sections = [`## ${role}`, visible.trim() || "_(empty message)_"];
    if (message.image_base64) sections.push("_[Image attachment]_ ");
    if (message.tool_calls?.length) {
      sections.push(
        "### Tool selections",
        ...message.tool_calls.map((call) =>
          `- **${call.name}**${call.arguments ? ` — \`${call.arguments.replace(/`/g, "\\`")}\`` : ""}`
        )
      );
    }
    return sections.join("\n\n");
  });
  return [`# ${title}`, "", ...body].join("\n\n").trimEnd() + "\n";
}

export function downloadConversation(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

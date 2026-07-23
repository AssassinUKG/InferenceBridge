import type { SessionInfo } from "./types";

const AUTOMATIC_CHAT_NAME = /^Chat ([1-9]\d*)$/;

/**
 * Returns the next automatic chat title from the current automatic-chat count.
 * The database compacts automatic titles after deletion, so this is only a UI
 * hint; the database remains the source of truth for the final number.
 */
export function nextAutomaticChatName(
  sessions: ReadonlyArray<Pick<SessionInfo, "name">>,
) {
  const automaticCount = sessions.reduce(
    (count, session) => count + (AUTOMATIC_CHAT_NAME.test(session.name ?? "") ? 1 : 0),
    0,
  );

  return `Chat ${automaticCount + 1}`;
}

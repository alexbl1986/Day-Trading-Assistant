import { Client, type Thread } from "@langchain/langgraph-sdk";

// Coat-check identity: the username is a key, not a
// credential. It owns threads (via thread metadata) and the per-user policy
// record; the portfolio/report data is baked in and shared by design.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "/api";

const STORAGE_KEY = "desk-username";

export function normalizeUsername(raw: string): string {
  return raw.trim().toLowerCase();
}

export function storedUsername(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(STORAGE_KEY);
}

export function storeUsername(username: string) {
  window.localStorage.setItem(STORAGE_KEY, username);
}

export function clearUsername() {
  window.localStorage.removeItem(STORAGE_KEY);
}

function client(): Client {
  return new Client({ apiUrl: API_URL });
}

function lastActive(thread: Thread): number {
  return Date.parse(thread.updated_at ?? thread.created_at) || 0;
}

/** Returning usernames land on their most recent conversation; new ones get
 *  a fresh thread stamped with their name. */
export async function resumeOrCreateThread(username: string): Promise<string> {
  const threads = await client().threads.search({
    metadata: { owner: username },
    limit: 50,
  });
  if (threads.length > 0) {
    const latest = [...threads].sort((a, b) => lastActive(b) - lastActive(a))[0];
    return latest.thread_id;
  }
  return createThread(username);
}

/** "New chat": one more thread for this username; older ones stay on the
 *  server but the UI only ever mounts the latest. */
export async function createThread(username: string): Promise<string> {
  const thread = await client().threads.create({
    metadata: { owner: username },
  });
  return thread.thread_id;
}

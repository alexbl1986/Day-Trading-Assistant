"use client";

import { useEffect, useState } from "react";
import { CandlestickChart, LogOut, Plus, UserRound } from "lucide-react";

import { Chat } from "@/components/chat";
import { Login } from "@/components/login";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Loader } from "@/components/ai-elements/loader";
import {
  clearUsername,
  createThread,
  resumeOrCreateThread,
  storeUsername,
  storedUsername,
} from "@/lib/session";

const ASSISTANT_ID = "trading_assistant";

type Session = { user: string; threadId: string };

export default function Page() {
  const [session, setSession] = useState<Session | null>(null);
  const [booting, setBooting] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // A stored username auto-resumes its latest thread on revisit.
  useEffect(() => {
    const stored = storedUsername();
    if (!stored) {
      setBooting(false);
      return;
    }
    resumeOrCreateThread(stored)
      .then((threadId) => setSession({ user: stored, threadId }))
      .catch(() => clearUsername())
      .finally(() => setBooting(false));
  }, []);

  const login = async (user: string) => {
    setBusy(true);
    setError(null);
    try {
      const threadId = await resumeOrCreateThread(user);
      storeUsername(user);
      setSession({ user, threadId });
    } catch {
      setError("Could not reach the agent server. Is it running?");
    } finally {
      setBusy(false);
    }
  };

  const newChat = async () => {
    if (!session || busy) return;
    setBusy(true);
    try {
      const threadId = await createThread(session.user);
      setSession({ ...session, threadId });
    } finally {
      setBusy(false);
    }
  };

  const switchUser = () => {
    clearUsername();
    setSession(null);
  };

  return (
    <main className="flex h-dvh flex-col">
      <header className="border-b bg-background">
        <div className="mx-auto flex w-full max-w-3xl items-center gap-2 px-4 py-3">
          <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <CandlestickChart className="size-4" />
          </div>
          <div className="leading-tight">
            <p className="text-sm font-medium">Trading Desk Assistant</p>
            <p className="text-xs text-muted-foreground">LangGraph + Next.js</p>
          </div>

          {session && (
            <div className="ml-auto flex items-center gap-2">
              <Badge variant="secondary" className="gap-1.5">
                <UserRound className="size-3" />
                {session.user}
              </Badge>
              <Button size="sm" variant="outline" onClick={newChat} disabled={busy}>
                <Plus className="size-4" />
                New chat
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={switchUser}
                aria-label="Switch user"
              >
                <LogOut className="size-4" />
              </Button>
            </div>
          )}
        </div>
      </header>

      {booting ? (
        <div className="flex flex-1 items-center justify-center gap-2 text-sm text-muted-foreground">
          <Loader size={16} />
          Resuming your desk...
        </div>
      ) : session ? (
        <Chat
          key={session.threadId}
          assistantId={ASSISTANT_ID}
          threadId={session.threadId}
          userId={session.user}
        />
      ) : (
        <Login onLogin={login} busy={busy} error={error} />
      )}
    </main>
  );
}

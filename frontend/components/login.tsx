"use client";

import { useState } from "react";
import { UserRound } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { normalizeUsername } from "@/lib/session";

export function Login({
  onLogin,
  busy,
  error,
}: {
  onLogin: (username: string) => void;
  busy: boolean;
  error: string | null;
}) {
  const [name, setName] = useState("");
  const username = normalizeUsername(name);

  return (
    <div className="flex flex-1 items-center justify-center px-4">
      <Card className="w-full max-w-sm">
        <CardContent className="space-y-4">
          <div className="flex flex-col items-center gap-3 pt-2 text-center">
            <div className="flex size-12 items-center justify-center rounded-full bg-muted">
              <UserRound className="size-6 text-muted-foreground" />
            </div>
            <div className="space-y-1">
              <h1 className="text-lg font-medium">Who&apos;s trading?</h1>
              <p className="text-sm text-muted-foreground">
                Pick any name — no password. A returning name resumes its latest
                conversation and its own policy rules; a new name starts fresh.
              </p>
            </div>
          </div>

          <form
            className="space-y-3"
            onSubmit={(e) => {
              e.preventDefault();
              if (username && !busy) onLogin(username);
            }}
          >
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. alex"
              autoFocus
              disabled={busy}
              aria-label="Username"
            />
            <Button type="submit" className="w-full" disabled={busy || !username}>
              {busy ? "Looking up your desk..." : "Continue"}
            </Button>
          </form>

          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>
    </div>
  );
}

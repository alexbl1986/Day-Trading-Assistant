"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Loader2, SendIcon, SquareIcon } from "lucide-react";
import type { ComponentProps, FormEvent, KeyboardEvent } from "react";

export type PromptInputProps = ComponentProps<"form">;

export const PromptInput = ({ className, ...props }: PromptInputProps) => (
  <form
    className={cn(
      "flex w-full flex-col gap-2 rounded-2xl border bg-background p-2 shadow-sm transition-colors focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/30",
      className
    )}
    {...props}
  />
);

export type PromptInputTextareaProps = ComponentProps<"textarea"> & {
  onSubmit?: () => void;
};

export const PromptInputTextarea = ({
  className,
  onSubmit,
  onKeyDown,
  ...props
}: PromptInputTextareaProps) => {
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    onKeyDown?.(e);
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      onSubmit?.();
    }
  };

  return (
    <textarea
      className={cn(
        "field-sizing-content max-h-48 min-h-10 w-full resize-none bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted-foreground disabled:opacity-60",
        className
      )}
      onKeyDown={handleKeyDown}
      rows={1}
      {...props}
    />
  );
};

export type PromptInputToolbarProps = ComponentProps<"div">;

export const PromptInputToolbar = ({
  className,
  ...props
}: PromptInputToolbarProps) => (
  <div
    className={cn("flex items-center justify-between gap-2 px-1", className)}
    {...props}
  />
);

export type PromptInputSubmitProps = ComponentProps<typeof Button> & {
  status?: "ready" | "streaming";
};

export const PromptInputSubmit = ({
  className,
  status = "ready",
  children,
  ...props
}: PromptInputSubmitProps) => {
  let icon = <SendIcon className="size-4" />;
  if (status === "streaming") icon = <SquareIcon className="size-4" />;

  return (
    <Button
      className={cn("rounded-full", className)}
      size="icon"
      type="submit"
      {...props}
    >
      {children ?? icon}
    </Button>
  );
};

// Convenience: prevent default and forward the event.
export function handlePromptSubmit(
  e: FormEvent<HTMLFormElement>,
  submit: () => void
) {
  e.preventDefault();
  submit();
}

export { Loader2 as PromptInputLoader };

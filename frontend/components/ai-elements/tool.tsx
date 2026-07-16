"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { CheckCircleIcon, ChevronDownIcon, ClockIcon } from "lucide-react";
import type { ComponentProps, ReactNode } from "react";

export type ToolState = "running" | "completed";

export type ToolProps = ComponentProps<"details"> & {
  icon?: ReactNode;
  title: string;
  state?: ToolState;
};

/**
 * Collapsible tool-call display. Uses a native <details> so it needs no
 * extra collapsible primitive, styled to match AI Elements' Tool component.
 */
export const Tool = ({
  className,
  icon,
  title,
  state = "completed",
  children,
  ...props
}: ToolProps) => (
  <details
    className={cn(
      "group not-prose w-full overflow-hidden rounded-lg border bg-muted/30 text-sm",
      className
    )}
    {...props}
  >
    <summary className="flex cursor-pointer list-none items-center justify-between gap-4 p-3">
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground [&_svg]:size-4">
          {icon}
        </span>
        <span className="font-medium text-foreground">{title}</span>
        <ToolStatusBadge state={state} />
      </div>
      <ChevronDownIcon className="size-4 text-muted-foreground transition-transform group-open:rotate-180" />
    </summary>
    <div className="border-t p-3">{children}</div>
  </details>
);

const ToolStatusBadge = ({ state }: { state: ToolState }) => (
  <Badge className="gap-1.5 rounded-full text-xs" variant="secondary">
    {state === "running" ? (
      <ClockIcon className="size-3 animate-pulse" />
    ) : (
      <CheckCircleIcon className="size-3 text-green-600" />
    )}
    {state === "running" ? "Running" : "Result"}
  </Badge>
);

export type ToolOutputProps = ComponentProps<"pre">;

export const ToolOutput = ({ className, ...props }: ToolOutputProps) => (
  <pre
    className={cn(
      "max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-muted/50 p-3 text-xs text-muted-foreground",
      className
    )}
    {...props}
  />
);

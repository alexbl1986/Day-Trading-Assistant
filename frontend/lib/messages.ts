export function getMessageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === "string") return block;
        if (block && typeof block === "object" && "text" in block) {
          return String((block as { text?: unknown }).text ?? "");
        }
        return "";
      })
      .join("");
  }
  return "";
}

export function toolLabel(name?: string): string {
  switch (name) {
    case "search_desk_reviews":
      return "Desk reviews";
    case "get_market_quote":
      return "Market quotes";
    case "search_web":
      return "Web search";
    default:
      return name ?? "tool";
  }
}

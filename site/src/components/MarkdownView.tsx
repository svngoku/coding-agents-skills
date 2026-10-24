import { useEffect, useRef } from "react";
import { cn } from "cn";
import { highlightWithin } from "@/lib/highlight";
import { renderMermaidWithin } from "@/lib/mermaid";

type MarkdownViewProps = {
  html: string;
  className?: string;
};

/** Enhances pre-generated Markdown with highlighted code and Mermaid diagrams. */
export function MarkdownView({ html, className }: MarkdownViewProps) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = container.current;
    if (!root) return;
    highlightWithin(root);
    return renderMermaidWithin(root);
  }, [html]);

  return (
    <div
      ref={container}
      className={cn("md", className)}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

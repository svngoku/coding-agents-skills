import { useEffect, useRef } from "react";
import { cn } from "cn";
import { highlightWithin } from "@/lib/highlight";

type MarkdownViewProps = {
  html: string;
  className?: string;
};

/** Renders pre-generated Markdown HTML and colourises its code blocks once. */
export function MarkdownView({ html, className }: MarkdownViewProps) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (container.current) highlightWithin(container.current);
  }, [html]);

  return (
    <div
      ref={container}
      className={cn("md", className)}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

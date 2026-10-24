import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TooltipAction } from "@/components/TooltipAction";

type CopyButtonProps = {
  value: string;
  label: string;
  className?: string;
};

export function CopyButton({ value, label, className }: CopyButtonProps) {
  const [copiedValue, setCopiedValue] = useState<string | null>(null);
  const copied = copiedValue === value;
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopiedValue(value);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopiedValue(null), 1600);
    } catch {
      setCopiedValue(null);
    }
  }

  return (
    <span className="flex items-center gap-1">
      <TooltipAction label={copied ? "Copied" : "Copy command"}>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          className={className}
          aria-label={
            copied ? `${label} copied` : `Copy ${label.toLowerCase()}`
          }
          onClick={copy}
        >
          {copied ? (
            <Check className="size-3.5" />
          ) : (
            <Copy className="size-3.5" />
          )}
        </Button>
      </TooltipAction>
      <span className="sr-only" role="status" aria-live="polite">
        {copied ? "Copied to clipboard" : ""}
      </span>
    </span>
  );
}

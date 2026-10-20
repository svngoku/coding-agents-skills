import type { ReactElement } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

type TooltipActionProps = {
  label: string;
  children: ReactElement;
};

/** Wraps a single control so it gains a shadcn tooltip without an extra wrapper element. */
export function TooltipAction({ label, children }: TooltipActionProps) {
  return (
    <Tooltip>
      <TooltipTrigger render={children} />
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

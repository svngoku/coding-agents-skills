import { useId, useState, type ReactElement } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

type TooltipActionProps = {
  label: string;
  children: ReactElement;
};

/** Wraps a single control so it gains a shadcn tooltip without an extra wrapper element. */
export function TooltipAction({ label, children }: TooltipActionProps) {
  const id = useId();
  const [open, setOpen] = useState(false);

  return (
    <Tooltip open={open} onOpenChange={setOpen}>
      <TooltipTrigger
        render={children}
        aria-describedby={open ? id : undefined}
      />
      <TooltipContent id={id}>{label}</TooltipContent>
    </Tooltip>
  );
}

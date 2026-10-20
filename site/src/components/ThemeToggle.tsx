import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TooltipAction } from "@/components/TooltipAction";
import type { Theme } from "@/lib/theme";

type ThemeToggleProps = {
  theme: Theme;
  onToggle: () => void;
};

export function ThemeToggle({ theme, onToggle }: ThemeToggleProps) {
  const next = theme === "dark" ? "light" : "dark";
  return (
    <TooltipAction label={`Switch to ${next} theme`}>
      <Button
        type="button"
        variant="outline"
        size="icon"
        aria-label={`Switch to ${next} theme`}
        onClick={onToggle}
      >
        {theme === "dark" ? <Sun /> : <Moon />}
      </Button>
    </TooltipAction>
  );
}

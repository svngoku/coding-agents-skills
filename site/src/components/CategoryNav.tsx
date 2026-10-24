import { Layers3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getCategoryIcon } from "@/lib/categories";
import type { GalleryCategory } from "@/lib/types";
import { cn } from "cn";

type CategoryNavProps = {
  categories: GalleryCategory[];
  category: string | null;
  onChange: (category: string | null) => void;
  total: number;
  compact?: boolean;
};

export function CategoryNav({
  categories,
  category,
  onChange,
  total,
  compact = false,
}: CategoryNavProps) {
  const items = [
    { value: "all", label: "All topics", count: total },
    ...categories.map((item) => ({
      value: item.name,
      label: item.name,
      count: item.count,
    })),
  ];

  if (compact) {
    return (
      <Select
        items={items}
        value={category ?? "all"}
        onValueChange={(value) => {
          if (value) onChange(value === "all" ? null : value);
        }}
      >
        <SelectTrigger
          aria-label="Filter by topic"
          className="h-10 w-full bg-card"
        >
          <Layers3
            className="size-4 shrink-0 text-muted-foreground"
            aria-hidden="true"
          />
          <SelectValue className="min-w-0 flex-1 text-left" />
        </SelectTrigger>
        <SelectContent align="start">
          {items.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
              <span className="ml-auto pl-3 text-xs tabular-nums text-muted-foreground">
                {item.count}
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }

  return (
    <nav aria-label="Skill topics">
      <h3 className="mb-3 px-3 text-sm font-medium text-muted-foreground">
        Browse by topic
      </h3>
      <ul className="space-y-1">
        {items.map((item) => {
          const active = item.value === (category ?? "all");
          const Icon =
            item.value === "all" ? Layers3 : getCategoryIcon(item.value);
          return (
            <li key={item.value}>
              <Button
                type="button"
                variant="ghost"
                aria-pressed={active}
                onClick={() =>
                  onChange(item.value === "all" ? null : item.value)
                }
                className={cn(
                  "h-auto min-h-11 w-full justify-start gap-2.5 rounded-lg px-3 py-2.5 text-left text-[13px] font-normal",
                  active
                    ? "bg-primary/8 font-medium text-primary hover:bg-primary/12 hover:text-primary"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                <Icon className="size-4 shrink-0" aria-hidden="true" />
                <span className="min-w-0 flex-1 whitespace-normal">
                  {item.label}
                </span>
                <span className="text-xs tabular-nums">{item.count}</span>
              </Button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

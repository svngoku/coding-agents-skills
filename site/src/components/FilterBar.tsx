import { Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { GalleryCategory, SortKey, ViewMode } from "@/lib/types";

type FilterBarProps = {
  query: string;
  onQueryChange: (value: string) => void;
  category: string | null;
  onCategoryChange: (value: string | null) => void;
  categories: GalleryCategory[];
  sort: SortKey;
  onSortChange: (value: SortKey) => void;
  view: ViewMode;
  onViewChange: (value: ViewMode) => void;
  total: number;
  visible: number;
};

export function FilterBar({
  query,
  onQueryChange,
  category,
  onCategoryChange,
  categories,
  sort,
  onSortChange,
  view,
  onViewChange,
  total,
  visible,
}: FilterBarProps) {
  const filters = [{ name: "All", count: total }, ...categories];

  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative w-full sm:max-w-sm">
          <Search
            className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
            aria-hidden="true"
          />
          <Input
            type="search"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search skills, capabilities, references…"
            aria-label="Search skills"
            className="pl-8 pr-8"
          />
          {query ? (
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              aria-label="Clear search"
              className="absolute top-1/2 right-1 -translate-y-1/2"
              onClick={() => onQueryChange("")}
            >
              <X className="size-3.5" />
            </Button>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor="skill-sort" className="sr-only">
            Sort skills
          </label>
          <select
            id="skill-sort"
            value={sort}
            onChange={(event) => onSortChange(event.target.value as SortKey)}
            className="h-7 rounded-lg border border-input bg-background px-2 text-sm text-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
          >
            <option value="name">Sort: name</option>
            <option value="refs">Sort: most guides</option>
            <option value="lines">Sort: most docs</option>
            <option value="category">Sort: category</option>
          </select>

          <div
            role="group"
            aria-label="Switch between cards and matrix"
            className="flex items-center gap-0.5 rounded-lg border p-0.5"
          >
            <Button
              type="button"
              size="xs"
              variant={view === "cards" ? "secondary" : "ghost"}
              aria-pressed={view === "cards"}
              onClick={() => onViewChange("cards")}
            >
              Cards
            </Button>
            <Button
              type="button"
              size="xs"
              variant={view === "matrix" ? "secondary" : "ghost"}
              aria-pressed={view === "matrix"}
              onClick={() => onViewChange("matrix")}
            >
              Matrix
            </Button>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Filter by category">
        {filters.map((item) => {
          const isActive = item.name === "All" ? category === null : category === item.name;
          return (
            <Button
              key={item.name}
              type="button"
              size="sm"
              variant={isActive ? "default" : "outline"}
              aria-pressed={isActive}
              className="gap-1.5 rounded-full"
              onClick={() => onCategoryChange(item.name === "All" ? null : item.name)}
            >
              {item.name}
              <span
                className={
                  isActive
                    ? "text-xs tabular-nums opacity-80"
                    : "text-xs tabular-nums text-muted-foreground"
                }
              >
                {item.count}
              </span>
            </Button>
          );
        })}
        <p className="ml-auto text-sm text-muted-foreground" role="status" aria-live="polite">
          {visible === total ? `${total} skills` : `${visible} of ${total} skills`}
        </p>
      </div>
    </div>
  );
}

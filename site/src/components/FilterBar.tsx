import { Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { GalleryCategory, SortKey, ViewMode } from "@/lib/types";

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "name", label: "Sort: name" },
  { value: "refs", label: "Sort: most guides" },
  { value: "lines", label: "Sort: most docs" },
  { value: "category", label: "Sort: category" },
];

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
        <InputGroup className="w-full sm:max-w-sm">
          <InputGroupAddon>
            <Search aria-hidden="true" />
          </InputGroupAddon>
          <InputGroupInput
            type="search"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search skills, capabilities, references…"
            aria-label="Search skills"
          />
          {query ? (
            <InputGroupAddon align="inline-end">
              <InputGroupButton
                size="icon-xs"
                aria-label="Clear search"
                onClick={() => onQueryChange("")}
              >
                <X />
              </InputGroupButton>
            </InputGroupAddon>
          ) : null}
        </InputGroup>

        <div className="flex flex-wrap items-center gap-2">
          <Select
            items={SORT_OPTIONS}
            value={sort}
            onValueChange={(value) => {
              if (value) onSortChange(value as SortKey);
            }}
          >
            <SelectTrigger
              id="skill-sort"
              size="sm"
              aria-label="Sort skills"
              className="w-full min-w-40 sm:w-auto"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent align="start">
              {SORT_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <ToggleGroup
            value={[view]}
            onValueChange={(next) => {
              const [value] = next;
              if (value) onViewChange(value as ViewMode);
            }}
            aria-label="Switch between cards and matrix"
            className="rounded-lg border p-0.5"
          >
            <ToggleGroupItem value="cards" size="sm" className="px-2.5 text-xs">
              Cards
            </ToggleGroupItem>
            <ToggleGroupItem value="matrix" size="sm" className="px-2.5 text-xs">
              Matrix
            </ToggleGroupItem>
          </ToggleGroup>
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
                    ? "text-xs tabular-nums"
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

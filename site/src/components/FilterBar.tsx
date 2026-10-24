import {
  ArrowDownWideNarrow,
  LayoutGrid,
  Search,
  Table2,
  X,
} from "lucide-react";
import { CategoryNav } from "@/components/CategoryNav";
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
  { value: "name", label: "Name, A to Z" },
  { value: "refs", label: "Most guides" },
  { value: "lines", label: "Most documentation" },
  { value: "category", label: "Category" },
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
  onReset: () => void;
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
  onReset,
}: FilterBarProps) {
  return (
    <div className="mb-5 space-y-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
        <InputGroup className="h-11 w-full bg-card sm:min-w-48 sm:flex-1">
          <InputGroupAddon>
            <Search aria-hidden="true" />
          </InputGroupAddon>
          <InputGroupInput
            type="search"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search skills, topics, or tools…"
            aria-label="Search skills"
            className="h-full text-base sm:text-sm [&::-webkit-search-cancel-button]:appearance-none"
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

        <div className="flex items-center gap-2">
          <Select
            items={SORT_OPTIONS}
            value={sort}
            onValueChange={(value) => {
              if (value) onSortChange(value as SortKey);
            }}
          >
            <SelectTrigger
              id="skill-sort"
              aria-label="Sort skills"
              className="min-w-0 flex-1 bg-card data-[size=default]:h-11 sm:w-44 sm:flex-none"
            >
              <ArrowDownWideNarrow
                className="size-3.5 shrink-0 text-muted-foreground"
                aria-hidden="true"
              />
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
            className="h-11 shrink-0 gap-0.5 rounded-lg border bg-card p-1"
          >
            <ToggleGroupItem
              value="cards"
              size="sm"
              aria-label="Card view"
              className="gap-1.5 rounded-md px-2.5 text-xs aria-pressed:bg-muted"
            >
              <LayoutGrid className="size-3.5" aria-hidden="true" />
              <span className="hidden xl:inline">Cards</span>
            </ToggleGroupItem>
            <ToggleGroupItem
              value="matrix"
              size="sm"
              aria-label="Matrix view"
              className="gap-1.5 rounded-md px-2.5 text-xs aria-pressed:bg-muted"
            >
              <Table2 className="size-3.5" aria-hidden="true" />
              <span className="hidden xl:inline">Matrix</span>
            </ToggleGroupItem>
          </ToggleGroup>
        </div>
      </div>

      <div className="lg:hidden">
        <CategoryNav
          compact
          categories={categories}
          category={category}
          onChange={onCategoryChange}
          total={total}
        />
      </div>

      <div className="flex min-h-8 flex-wrap items-center justify-between gap-2">
        <p
          className="text-xs text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          {visible === total
            ? `Showing all ${total} skills`
            : `Showing ${visible} of ${total} skills`}
          {category ? ` in ${category}` : ""}
        </p>
        {query || category ? (
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={onReset}
            className="text-muted-foreground"
          >
            <X className="size-3" aria-hidden="true" />
            Reset filters
          </Button>
        ) : (
          <span className="hidden text-xs text-muted-foreground sm:inline">
            Read first. Install what you need.
          </span>
        )}
      </div>
    </div>
  );
}

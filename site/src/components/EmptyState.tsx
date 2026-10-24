import { SearchX } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";

type EmptyStateProps = {
  onReset: () => void;
};

export function EmptyState({ onReset }: EmptyStateProps) {
  return (
    <Empty className="min-h-80 rounded-2xl border border-dashed bg-card p-8">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <SearchX aria-hidden="true" />
        </EmptyMedia>
        <EmptyTitle className="text-base font-semibold">
          No skills match those filters
        </EmptyTitle>
        <EmptyDescription className="text-pretty">
          Try another search, or reset the filters to browse the full library.
        </EmptyDescription>
      </EmptyHeader>
      <EmptyContent>
        <Button type="button" variant="outline" size="sm" onClick={onReset}>
          Clear filters
        </Button>
      </EmptyContent>
    </Empty>
  );
}

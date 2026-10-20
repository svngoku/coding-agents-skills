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
    <Empty className="border bg-card">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <SearchX aria-hidden="true" />
        </EmptyMedia>
        <EmptyTitle className="text-base font-semibold">No skills match those filters</EmptyTitle>
        <EmptyDescription className="text-pretty">
          Try a different term, or clear the category filter to see all fifteen skills.
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

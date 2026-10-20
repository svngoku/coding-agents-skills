import { SearchX } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

type EmptyStateProps = {
  onReset: () => void;
};

export function EmptyState({ onReset }: EmptyStateProps) {
  return (
    <Card className="items-center py-10 text-center">
      <CardContent className="flex max-w-md flex-col items-center gap-3 px-6">
        <span className="grid size-10 place-items-center rounded-full bg-muted">
          <SearchX className="size-5 text-muted-foreground" aria-hidden="true" />
        </span>
        <div className="space-y-1">
          <h2 className="text-base font-semibold">No skills match those filters</h2>
          <p className="text-pretty text-sm text-muted-foreground">
            Try a different term, or clear the category filter to see all fifteen skills.
          </p>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={onReset}>
          Clear filters
        </Button>
      </CardContent>
    </Card>
  );
}

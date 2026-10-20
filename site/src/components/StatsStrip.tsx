import type { GalleryStats } from "@/lib/types";

type StatsStripProps = {
  stats: GalleryStats;
};

const number = new Intl.NumberFormat("en-US");

export function StatsStrip({ stats }: StatsStripProps) {
  const items = [
    { label: "Skills", value: number.format(stats.skills), hint: "loadable agent skills" },
    { label: "Reference guides", value: number.format(stats.references), hint: "deep-dive documents" },
    { label: "Eval tasks", value: number.format(stats.evalTasks), hint: "skillgrade harnesses" },
    { label: "Example scripts", value: number.format(stats.scripts), hint: "runnable helpers" },
    { label: "Documentation", value: number.format(stats.docsLines), hint: "markdown lines" },
  ];

  return (
    <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border bg-border sm:grid-cols-3 lg:grid-cols-5">
      {items.map((item) => (
        <div key={item.label} className="bg-card px-4 py-3">
          <dt className="text-xs font-medium text-muted-foreground">{item.label}</dt>
          <dd className="mt-0.5 text-2xl font-semibold tabular-nums text-card-foreground">
            {item.value}
          </dd>
          <dd className="text-xs text-muted-foreground">{item.hint}</dd>
        </div>
      ))}
    </dl>
  );
}

import type { GalleryStats } from "@/lib/types";

type StatsStripProps = {
  stats: GalleryStats;
};

const number = new Intl.NumberFormat("en-US");

export function StatsStrip({ stats }: StatsStripProps) {
  const items = [
    { label: "Agent skills", value: number.format(stats.skills) },
    { label: "Reference guides", value: number.format(stats.references) },
    { label: "Eval tasks", value: number.format(stats.evalTasks) },
  ];

  return (
    <dl className="grid grid-cols-3 gap-4 border-t pt-4">
      {items.map((item) => (
        <div key={item.label} className="flex flex-col gap-1">
          <dt className="order-2 text-xs text-muted-foreground">
            {item.label}
          </dt>
          <dd className="font-heading text-[1.4rem] leading-tight font-medium tracking-tight tabular-nums">
            {item.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

import { useState } from "react";
import { cn } from "cn";
import { getCategoryIcon } from "@/lib/categories";
import type { Skill } from "@/lib/types";

// Match bundled marks to skill names. Unbranded skills keep their topic icon.
const marks = import.meta.glob<string>("../assets/brands/*.{svg,png}", {
  eager: true,
  import: "default",
  query: "?url",
});

type SkillIconProps = {
  skill: Pick<Skill, "name" | "category">;
};

export function SkillIcon({ skill }: SkillIconProps) {
  const source =
    marks[`../assets/brands/${skill.name}.svg`] ??
    marks[`../assets/brands/${skill.name}.png`];
  const [failedSource, setFailedSource] = useState<string | null>(null);
  const showMark = source && failedSource !== source;
  const Icon = getCategoryIcon(skill.category);

  return (
    <span
      aria-hidden="true"
      data-slot="skill-icon"
      data-brand={showMark ? skill.name : undefined}
      className={cn(
        "flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-lg text-foreground",
        showMark ? "ring-1 ring-black/5" : "bg-muted",
        showMark && (skill.name === "prime-rl" ? "bg-black" : "bg-white"),
      )}
    >
      {showMark ? (
        <img
          src={source}
          alt=""
          width={24}
          height={24}
          decoding="async"
          className="size-6 object-contain"
          onError={() => setFailedSource(source)}
        />
      ) : (
        <Icon className="size-4" aria-hidden="true" />
      )}
    </span>
  );
}

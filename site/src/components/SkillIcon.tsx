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

// The five official Linkup skills share their provider's mark.
const sharedMarkNames: Readonly<Record<string, string>> = {
  "linkup-extract": "linkup",
  "linkup-fetch": "linkup",
  "linkup-research": "linkup",
  "linkup-search": "linkup",
  "linkup-workflow": "linkup",
};

type SkillIconProps = {
  skill: Pick<Skill, "name" | "category">;
};

export function SkillIcon({ skill }: SkillIconProps) {
  const markName = sharedMarkNames[skill.name] ?? skill.name;
  const source =
    marks[`../assets/brands/${markName}.svg`] ??
    marks[`../assets/brands/${markName}.png`];
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

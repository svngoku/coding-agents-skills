import {
  BookOpen,
  Check,
  ChevronRight,
  FileCode2,
  Files,
  FolderOpen,
} from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TooltipAction } from "@/components/TooltipAction";
import { cn } from "cn";
import { getCategoryIcon } from "@/lib/categories";
import type { Skill } from "@/lib/types";

type SkillCardProps = {
  skill: Skill;
  onDetails: (skill: Skill) => void;
  onRead: (skill: Skill) => void;
};

const number = new Intl.NumberFormat("en-US");

export function SkillCard({ skill, onDetails, onRead }: SkillCardProps) {
  const Icon = getCategoryIcon(skill.category);

  return (
    <Card
      id={`skill-${skill.name}`}
      className="skill-card h-full min-w-0 scroll-mt-8 gap-0 rounded-2xl border bg-card p-0 ring-0 transition-colors hover:border-foreground/25"
    >
      <CardHeader className="gap-4 p-5 pb-3">
        <div className="flex min-h-8 items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
            <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted text-foreground">
              <Icon className="size-4" aria-hidden="true" />
            </span>
            <span>{skill.category}</span>
          </div>
          <TooltipAction label="Skill directory on GitHub">
            <a
              href={skill.dirUrl}
              target="_blank"
              rel="noreferrer"
              aria-label={`${skill.title} directory on GitHub`}
              className={cn(
                buttonVariants({ variant: "ghost", size: "icon-sm" }),
                "shrink-0 text-muted-foreground",
              )}
            >
              <FolderOpen className="size-4" aria-hidden="true" />
            </a>
          </TooltipAction>
        </div>
        <div className="min-w-0">
          <CardTitle>
            <h3>
              <button
                type="button"
                onClick={() => onDetails(skill)}
                className="min-h-13 w-full text-left text-[1.2rem] leading-[1.35] font-medium tracking-tight text-balance hover:text-primary"
              >
                {skill.title}
              </button>
            </h3>
          </CardTitle>
          <CardDescription className="mt-1.5">
            <code className="text-[11px] break-all">{skill.name}</code>
          </CardDescription>
        </div>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col gap-4 px-5 pb-5">
        <p className="line-clamp-3 min-h-[4.5rem] text-sm leading-6 text-pretty text-muted-foreground">
          {skill.description}
        </p>

        <ul
          className="min-h-11 space-y-1.5"
          aria-label={`${skill.title} capabilities`}
        >
          {skill.capabilities.slice(0, 2).map((capability) => (
            <li
              key={capability}
              className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground"
              title={capability}
            >
              <Check
                className="size-3 shrink-0 text-primary"
                aria-hidden="true"
              />
              <span className="truncate">{capability}</span>
            </li>
          ))}
        </ul>

        <div className="mt-auto flex flex-wrap items-center gap-x-4 gap-y-2 text-xs tabular-nums text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <BookOpen className="size-3.5" aria-hidden="true" />
            {skill.references.length}{" "}
            {skill.references.length === 1 ? "guide" : "guides"}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <Files className="size-3.5" aria-hidden="true" />
            {number.format(skill.docsLines)} lines
          </span>
        </div>
      </CardContent>

      <CardFooter className="justify-between gap-1 rounded-b-2xl bg-muted/25 px-4 py-3">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => onDetails(skill)}
        >
          Details
          <ChevronRight className="size-3.5" aria-hidden="true" />
        </Button>
        <div className="flex items-center gap-1">
          <TooltipAction label="SKILL.md on GitHub">
            <a
              href={skill.skillUrl}
              target="_blank"
              rel="noreferrer"
              aria-label={`${skill.title} SKILL.md on GitHub`}
              className={cn(
                buttonVariants({ variant: "ghost", size: "icon-sm" }),
                "text-muted-foreground",
              )}
            >
              <FileCode2 className="size-3.5" aria-hidden="true" />
            </a>
          </TooltipAction>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => onRead(skill)}
          >
            <BookOpen className="size-3.5" aria-hidden="true" />
            Read skill
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
}

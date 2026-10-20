import { BookOpen, FileCode2, FileText, FolderOpen } from "lucide-react";
import { Badge } from "@/components/ui/badge";
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
import type { Skill } from "@/lib/types";

type SkillCardProps = {
  skill: Skill;
  onDetails: (skill: Skill) => void;
  onRead: (skill: Skill) => void;
};

const CAPABILITY_BUDGET = 64;

function visibleCapabilities(capabilities: string[]) {
  const visible: string[] = [];
  let used = 0;
  for (const capability of capabilities) {
    if (visible.length >= 3) break;
    if (visible.length > 0 && used + capability.length > CAPABILITY_BUDGET) break;
    visible.push(capability);
    used += capability.length;
  }
  return { visible, hidden: capabilities.length - visible.length };
}

export function SkillCard({ skill, onDetails, onRead }: SkillCardProps) {
  const { visible, hidden } = visibleCapabilities(skill.capabilities);
  const evalLabel = skill.eval.present
    ? `${skill.eval.tasks.length} eval task${skill.eval.tasks.length === 1 ? "" : "s"}`
    : "no eval harness";

  return (
    <Card
      id={`skill-${skill.name}`}
      className="h-full scroll-mt-24 gap-3 transition-colors hover:ring-foreground/25"
    >
      <CardHeader className="gap-1.5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <CardTitle className="text-base leading-snug font-semibold">
              <button
                type="button"
                onClick={() => onDetails(skill)}
                className="line-clamp-2 min-h-[2.75rem] text-left text-balance transition-colors hover:text-primary"
              >
                {skill.title}
              </button>
            </CardTitle>
            <CardDescription className="truncate font-mono text-xs">
              skills/{skill.name}/SKILL.md
            </CardDescription>
          </div>
          <Badge variant="secondary" className="shrink-0">
            {skill.category}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col gap-3">
        <p className="line-clamp-3 min-h-[4.5rem] text-pretty text-sm leading-6 text-muted-foreground">
          {skill.description}
        </p>

        <ul
          className="flex min-h-14 max-h-14 flex-wrap content-start gap-1.5 overflow-hidden"
          aria-label={`${skill.title} capabilities`}
        >
          {visible.map((capability) => (
            <li
              key={capability}
              className="max-w-full truncate rounded-md border px-2 py-0.5 text-xs text-muted-foreground"
              title={capability}
            >
              {capability}
            </li>
          ))}
          {hidden > 0 ? (
            <li className="rounded-md border border-dashed px-2 py-0.5 text-xs tabular-nums text-muted-foreground">
              +{hidden} more
            </li>
          ) : null}
          {skill.capabilities.length === 0 ? (
            <li className="text-xs text-muted-foreground">No capability tags</li>
          ) : null}
        </ul>

        <div className="mt-auto flex flex-wrap items-center gap-1.5">
          <Badge variant="outline" className="tabular-nums">
            {skill.references.length} guides
          </Badge>
          <Badge variant="outline" className="tabular-nums">
            {skill.docsLines} lines
          </Badge>
          <Badge variant="secondary" className="hidden sm:inline-flex">
            {skill.status}
          </Badge>
          <Badge variant="outline" className="hidden sm:inline-flex">
            {evalLabel}
          </Badge>
        </div>
      </CardContent>

      <CardFooter className="justify-between gap-2 bg-transparent">
        <div className="flex items-center gap-0.5">
          <Button type="button" variant="ghost" size="sm" onClick={() => onDetails(skill)}>
            <FileText />
            Details
          </Button>
          <Button type="button" variant="ghost" size="sm" onClick={() => onRead(skill)}>
            <BookOpen />
            Read
          </Button>
        </div>
        <div className="flex items-center gap-0.5">
          <TooltipAction label="SKILL.md on GitHub">
            <a
              href={skill.skillUrl}
              target="_blank"
              rel="noreferrer"
              aria-label={`${skill.title} SKILL.md on GitHub`}
              className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }))}
            >
              <FileCode2 className="size-3.5" />
            </a>
          </TooltipAction>
          <TooltipAction label="Skill directory on GitHub">
            <a
              href={skill.dirUrl}
              target="_blank"
              rel="noreferrer"
              aria-label={`${skill.title} directory on GitHub`}
              className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }))}
            >
              <FolderOpen className="size-3.5" />
            </a>
          </TooltipAction>
        </div>
      </CardFooter>
    </Card>
  );
}

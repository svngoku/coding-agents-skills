import { useState } from "react";
import {
  BookOpen,
  Check,
  ChevronRight,
  FileCode2,
  FileText,
  FolderOpen,
  Terminal,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { TooltipAction } from "@/components/TooltipAction";
import { cn } from "cn";
import type { GalleryDocument, Skill } from "@/lib/types";
import { getCategoryIcon } from "@/lib/categories";

type SkillDetailsDialogProps = {
  skill: Skill | null;
  onClose: () => void;
  onRead: (skill: Skill) => void;
  onOpenDocument: (skill: Skill, document: GalleryDocument) => void;
};

type Section = "references" | "scripts" | "overview";

export function SkillDetailsDialog({
  skill,
  onClose,
  onRead,
  onOpenDocument,
}: SkillDetailsDialogProps) {
  const [section, setSection] = useState<Section>(() =>
    skill?.references.length
      ? "references"
      : skill?.scripts.length
        ? "scripts"
        : "overview",
  );
  const sections: { id: Section; label: string; count?: number }[] = skill
    ? [
        { id: "references", label: "Guides", count: skill.references.length },
        { id: "scripts", label: "Scripts", count: skill.scripts.length },
        { id: "overview", label: "Overview" },
      ]
    : [];
  const Icon = getCategoryIcon(skill?.category ?? "");

  return (
    <Dialog
      open={skill !== null}
      onOpenChange={(open) => (open ? undefined : onClose())}
    >
      {skill ? (
        <DialogContent
          key={skill.name}
          className="max-h-[92dvh] w-[calc(100vw-2rem)] max-w-2xl grid-rows-[auto_minmax(0,1fr)_auto] gap-0 overflow-hidden rounded-2xl p-0 sm:max-w-2xl"
        >
          <DialogHeader className="gap-2 border-b bg-popover p-5 pr-12 sm:p-6 sm:pr-12">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Icon className="size-4 shrink-0" aria-hidden="true" />
              <span>{skill.category}</span>
            </div>
            <DialogTitle className="text-xl leading-snug font-medium tracking-tight sm:text-2xl">
              {skill.title}
            </DialogTitle>
            <DialogDescription className="font-mono text-[11px] break-all">
              skills/{skill.name}/
            </DialogDescription>
          </DialogHeader>

          <div
            data-slot="details-scroll"
            className="min-h-0 space-y-5 overflow-y-auto p-5 sm:p-6"
          >
            <p className="text-sm leading-6 text-pretty text-muted-foreground">
              {skill.description}
            </p>
            <dl className="grid grid-cols-3 gap-3 rounded-xl bg-muted/60 p-4">
              <div>
                <dt className="text-[11px] text-muted-foreground">Docs</dt>
                <dd className="mt-1 text-sm font-medium tabular-nums">
                  {new Intl.NumberFormat("en-US").format(skill.docsLines)} lines
                </dd>
              </div>
              <div>
                <dt className="text-[11px] text-muted-foreground">
                  Evaluation
                </dt>
                <dd className="mt-1 text-sm font-medium tabular-nums">
                  {skill.eval.present
                    ? `${skill.eval.tasks.length} ${skill.eval.tasks.length === 1 ? "task" : "tasks"}`
                    : "No harness"}
                </dd>
              </div>
              <div>
                <dt className="text-[11px] text-muted-foreground">Status</dt>
                <dd className="mt-1">
                  <Badge variant="secondary" className="text-primary">
                    {skill.status}
                  </Badge>
                </dd>
              </div>
            </dl>
            <ToggleGroup
              value={[section]}
              onValueChange={(next) => {
                const [value] = next;
                if (value) setSection(value as Section);
              }}
              aria-label="Skill details"
              className="w-full flex-wrap gap-1 rounded-lg bg-muted p-1"
            >
              {sections.map((item) => (
                <ToggleGroupItem
                  key={item.id}
                  value={item.id}
                  size="sm"
                  className="min-w-0 flex-1 basis-16 justify-center gap-1.5 rounded-md px-2 text-xs aria-pressed:bg-card aria-pressed:text-foreground aria-pressed:shadow-xs sm:text-sm"
                >
                  {item.label}
                  {typeof item.count === "number" ? (
                    <span className="text-xs tabular-nums text-muted-foreground">
                      {item.count}
                    </span>
                  ) : null}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>

            {section === "references" ? (
              <div>
                {skill.references.length ? (
                  <ul className="divide-y">
                    {skill.references.map((reference) => {
                      const path = `references/${reference.file}`;
                      const document = skill.documents.find(
                        (item) => item.path === path,
                      );
                      return (
                        <li
                          key={reference.file}
                          className="flex items-center gap-1 py-1"
                        >
                          <Button
                            type="button"
                            variant="ghost"
                            disabled={!document}
                            onClick={() =>
                              document && onOpenDocument(skill, document)
                            }
                            className="h-auto min-w-0 flex-1 items-start justify-start gap-3 px-2 py-3 text-left whitespace-normal"
                          >
                            <FileText
                              className="mt-1 size-4 shrink-0 text-muted-foreground"
                              aria-hidden="true"
                            />
                            <span className="min-w-0 flex-1">
                              <span className="block text-sm leading-6 font-medium [overflow-wrap:anywhere]">
                                {reference.title}
                              </span>
                              <span className="mt-1 block text-xs font-normal tabular-nums text-muted-foreground">
                                {reference.lines} lines
                              </span>
                            </span>
                            <ChevronRight
                              className="mt-1 hidden size-3.5 shrink-0 text-muted-foreground sm:block"
                              aria-hidden="true"
                            />
                          </Button>
                          <TooltipAction label="Open on GitHub">
                            <a
                              href={reference.url}
                              target="_blank"
                              rel="noreferrer"
                              className={cn(
                                buttonVariants({
                                  variant: "ghost",
                                  size: "icon-sm",
                                }),
                                "shrink-0",
                              )}
                              aria-label={`${reference.title} on GitHub`}
                            >
                              <FileCode2 className="size-3.5" />
                            </a>
                          </TooltipAction>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <Empty className="p-5">
                    <EmptyHeader>
                      <EmptyMedia variant="icon">
                        <BookOpen aria-hidden="true" />
                      </EmptyMedia>
                      <EmptyTitle className="text-sm">
                        Everything is in the skill
                      </EmptyTitle>
                      <EmptyDescription className="text-xs">
                        There are no separate guides. Open SKILL.md to read the
                        complete guidance.
                      </EmptyDescription>
                    </EmptyHeader>
                  </Empty>
                )}
              </div>
            ) : null}

            {section === "scripts" ? (
              <div>
                {skill.scripts.length ? (
                  <ul className="divide-y">
                    {skill.scripts.map((script) => {
                      const document = skill.documents.find(
                        (item) => item.path === script.file,
                      );
                      return (
                        <li key={script.file} className="py-1">
                          <Button
                            type="button"
                            variant="ghost"
                            disabled={!document}
                            onClick={() =>
                              document && onOpenDocument(skill, document)
                            }
                            className="h-auto w-full items-start justify-start gap-3 px-2 py-3 text-left whitespace-normal"
                          >
                            <Terminal
                              className="mt-1 size-4 shrink-0 text-muted-foreground"
                              aria-hidden="true"
                            />
                            <span className="min-w-0 flex-1">
                              <code className="block text-xs leading-6 break-all">
                                {script.file}
                              </code>
                              <span className="mt-1 block text-xs font-normal tabular-nums text-muted-foreground">
                                {script.lines} lines
                              </span>
                            </span>
                            <ChevronRight
                              className="mt-1 size-3.5 shrink-0 text-muted-foreground"
                              aria-hidden="true"
                            />
                          </Button>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <Empty className="p-5">
                    <EmptyHeader>
                      <EmptyMedia variant="icon">
                        <Terminal aria-hidden="true" />
                      </EmptyMedia>
                      <EmptyTitle className="text-sm">
                        No separate scripts
                      </EmptyTitle>
                      <EmptyDescription className="text-xs">
                        Code examples are included in the skill and its guides.
                      </EmptyDescription>
                    </EmptyHeader>
                  </Empty>
                )}
              </div>
            ) : null}

            {section === "overview" ? (
              <div className="space-y-6">
                {skill.capabilities.length ? (
                  <section aria-label="Capabilities">
                    <h3 className="mb-3 text-sm font-medium">
                      What this skill covers
                    </h3>
                    <ul className="space-y-3">
                      {skill.capabilities.map((capability) => (
                        <li
                          key={capability}
                          className="flex items-start gap-2.5 text-sm leading-6 text-muted-foreground"
                        >
                          <Check
                            className="mt-1.5 size-3 shrink-0 text-primary"
                            aria-hidden="true"
                          />
                          <span className="min-w-0 [overflow-wrap:anywhere]">
                            {capability}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </section>
                ) : null}
                {skill.useCases.length ? (
                  <section aria-label="Use cases">
                    <h3 className="mb-3 text-sm font-medium">When to use it</h3>
                    <ul className="list-disc space-y-2 pl-5 text-sm leading-6 text-muted-foreground">
                      {skill.useCases.map((useCase) => (
                        <li key={useCase}>{useCase}</li>
                      ))}
                    </ul>
                  </section>
                ) : null}
              </div>
            ) : null}
          </div>

          <DialogFooter className="m-0 flex-row flex-wrap items-center justify-between gap-2 rounded-b-2xl bg-popover px-5 py-4 sm:justify-between">
            <Button type="button" size="sm" onClick={() => onRead(skill)}>
              <BookOpen />
              Read skill
            </Button>
            <TooltipAction label="Skill directory on GitHub">
              <a
                href={skill.dirUrl}
                target="_blank"
                rel="noreferrer"
                aria-label={`${skill.title} directory on GitHub`}
                className={cn(
                  buttonVariants({ variant: "ghost", size: "icon-sm" }),
                  "ml-auto text-muted-foreground",
                )}
              >
                <FolderOpen />
              </a>
            </TooltipAction>
            <a
              href={skill.skillUrl}
              target="_blank"
              rel="noreferrer"
              className={cn(buttonVariants({ variant: "outline", size: "sm" }))}
            >
              <FileCode2 />
              Source
            </a>
          </DialogFooter>
        </DialogContent>
      ) : null}
    </Dialog>
  );
}

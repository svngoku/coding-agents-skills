import { useState } from "react";
import { BookOpen, FileCode2, FolderOpen } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "cn";
import type { GalleryDocument, Skill } from "@/lib/types";

type SkillDetailsDialogProps = {
  skill: Skill | null;
  onClose: () => void;
  onRead: (skill: Skill) => void;
  onOpenDocument: (skill: Skill, document: GalleryDocument) => void;
};

type Section = "references" | "scripts" | "use-cases";

export function SkillDetailsDialog({
  skill,
  onClose,
  onRead,
  onOpenDocument,
}: SkillDetailsDialogProps) {
  const [section, setSection] = useState<Section>("references");
  const sections: { id: Section; label: string; count?: number }[] = skill
    ? [
        { id: "references", label: "Guides", count: skill.references.length },
        { id: "scripts", label: "Scripts", count: skill.scripts.length },
        { id: "use-cases", label: "Use cases" },
      ]
    : [];

  return (
    <Dialog open={skill !== null} onOpenChange={(open) => (open ? undefined : onClose())}>
      {skill ? (
        <DialogContent key={skill.name} className="flex max-h-[90dvh] w-[calc(100vw-2rem)] max-w-2xl flex-col gap-0 p-0 sm:max-w-2xl">
          <DialogHeader className="gap-1 border-b p-5 pr-12">
            <DialogTitle className="text-lg">{skill.title}</DialogTitle>
            <DialogDescription className="flex flex-wrap items-center gap-2 font-mono text-xs">
              skills/{skill.name}/SKILL.md
              <Badge variant="secondary" className="font-sans">
                {skill.category}
              </Badge>
            </DialogDescription>
            <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
              <Badge variant="outline">{skill.status}</Badge>
              <Badge variant="outline" className="tabular-nums">
                {skill.docsLines} lines
              </Badge>
              <Badge variant="outline">
                {skill.eval.present
                  ? `${skill.eval.tasks.length} eval task${skill.eval.tasks.length === 1 ? "" : "s"}`
                  : "no eval harness"}
              </Badge>
            </div>
          </DialogHeader>

          <div className="min-h-0 flex-1 overflow-y-auto p-5">
            <div
              role="group"
              aria-label="Skill details"
              className="flex flex-wrap gap-1 rounded-lg bg-muted p-1"
            >
              {sections.map((item) => {
                const isActive = section === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    aria-pressed={isActive}
                    onClick={() => setSection(item.id)}
                    className={cn(
                      "inline-flex h-7 flex-1 items-center justify-center gap-1.5 rounded-md px-2 text-sm font-medium whitespace-nowrap transition-colors",
                      isActive
                        ? "bg-background text-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {item.label}
                    {typeof item.count === "number" ? (
                      <span className="text-xs tabular-nums opacity-70">{item.count}</span>
                    ) : null}
                  </button>
                );
              })}
            </div>

            {section === "references" ? (
              <div className="mt-3">
                {skill.references.length ? (
                  <ul className="space-y-2">
                    {skill.references.map((reference) => {
                      const path = `references/${reference.file}`;
                      const document = skill.documents.find((item) => item.path === path);
                      return (
                        <li key={reference.file} className="flex items-start gap-2 rounded-lg border p-2.5">
                          <div className="min-w-0 flex-1">
                            <button
                              type="button"
                              onClick={() => document && onOpenDocument(skill, document)}
                              className="block w-full text-left text-sm leading-snug font-medium text-balance hover:text-primary"
                            >
                              {reference.title}
                            </button>
                            <p className="mt-1 text-xs tabular-nums text-muted-foreground">
                              {reference.lines} lines
                            </p>
                          </div>
                          <a
                            href={reference.url}
                            target="_blank"
                            rel="noreferrer"
                            className={cn(buttonVariants({ variant: "ghost", size: "icon-xs" }), "shrink-0")}
                            aria-label={`${reference.title} on GitHub`}
                            title="Open on GitHub"
                          >
                            <FileCode2 className="size-3.5" />
                          </a>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">This skill ships no reference guides.</p>
                )}
              </div>
            ) : null}

            {section === "scripts" ? (
              <div className="mt-3">
                {skill.scripts.length ? (
                  <ul className="space-y-2">
                    {skill.scripts.map((script) => {
                      const document = skill.documents.find((item) => item.path === script.file);
                      return (
                        <li key={script.file} className="rounded-lg border p-2.5">
                          <button
                            type="button"
                            onClick={() => document && onOpenDocument(skill, document)}
                            className="block w-full text-left font-mono text-xs leading-snug font-medium break-all hover:text-primary"
                          >
                            {script.file}
                          </button>
                          <p className="mt-1 text-xs tabular-nums text-muted-foreground">
                            {script.lines} lines
                          </p>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">This skill ships no example scripts.</p>
                )}
              </div>
            ) : null}

            {section === "use-cases" ? (
              <div className="mt-3">
                {skill.useCases.length ? (
                  <ul className="list-disc space-y-2 pl-5 text-sm text-muted-foreground">
                    {skill.useCases.map((useCase) => (
                      <li key={useCase}>{useCase}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">No use cases listed in the README.</p>
                )}
              </div>
            ) : null}
          </div>

          <div className="flex flex-wrap items-center gap-2 border-t p-4">
            <Button type="button" size="sm" onClick={() => onRead(skill)}>
              <BookOpen />
              Read the skill
            </Button>
            <a
              href={skill.skillUrl}
              target="_blank"
              rel="noreferrer"
              className={cn(buttonVariants({ variant: "outline", size: "sm" }))}
            >
              <FileCode2 />
              SKILL.md
            </a>
            <a
              href={skill.dirUrl}
              target="_blank"
              rel="noreferrer"
              className={cn(buttonVariants({ variant: "outline", size: "sm" }))}
            >
              <FolderOpen />
              Directory
            </a>
          </div>
        </DialogContent>
      ) : null}
    </Dialog>
  );
}

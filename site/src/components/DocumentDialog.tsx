import { ExternalLink } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { MarkdownView } from "@/components/MarkdownView";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "cn";
import type { GalleryDocument, Skill } from "@/lib/types";

export type DocumentSelection = {
  skill: Skill;
  document: GalleryDocument;
};

type DocumentDialogProps = {
  selection: DocumentSelection | null;
  onClose: () => void;
  onSelect: (document: GalleryDocument) => void;
};

export function DocumentDialog({ selection, onClose, onSelect }: DocumentDialogProps) {
  return (
    <Dialog open={selection !== null} onOpenChange={(open) => (open ? undefined : onClose())}>
      {selection ? (
        <DialogContent className="flex h-[92dvh] w-[calc(100vw-2rem)] max-w-4xl flex-col gap-0 p-0 sm:max-w-4xl">
          <DialogHeader className="gap-1 border-b p-5 pr-12">
            <DialogTitle className="text-lg">{selection.document.label}</DialogTitle>
            <DialogDescription className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs">
              <span className="break-all">
                skills/{selection.skill.name}/{selection.document.path}
              </span>
              <span className="tabular-nums">{selection.document.lines} lines</span>
              <a
                href={selection.document.url}
                target="_blank"
                rel="noreferrer"
                className={cn(buttonVariants({ variant: "ghost", size: "xs" }), "font-sans")}
              >
                GitHub
                <ExternalLink className="size-3" aria-hidden="true" />
              </a>
            </DialogDescription>
            <div className="mt-2">
              <label htmlFor="document-switcher" className="sr-only">
                Switch document
              </label>
              <select
                id="document-switcher"
                value={selection.document.path}
                onChange={(event) => {
                  const next = selection.skill.documents.find(
                    (item) => item.path === event.target.value,
                  );
                  if (next) onSelect(next);
                }}
                className="h-8 w-full rounded-lg border border-input bg-background px-2 text-sm text-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
              >
                {(
                  [
                    { label: "Skill", kind: "skill" },
                    { label: "Reference guides", kind: "guide" },
                    { label: "Scripts", kind: "script" },
                  ] as const
                ).map((group) => {
                  const documents = selection.skill.documents.filter(
                    (item) => item.kind === group.kind,
                  );
                  if (!documents.length) return null;
                  return (
                    <optgroup key={group.kind} label={group.label}>
                      {documents.map((item) => (
                        <option key={item.path} value={item.path}>
                          {item.label}
                        </option>
                      ))}
                    </optgroup>
                  );
                })}
              </select>
            </div>
          </DialogHeader>
          <div className="min-h-0 flex-1 overflow-y-auto p-5">
            <MarkdownView html={selection.document.html} />
          </div>
        </DialogContent>
      ) : null}
    </Dialog>
  );
}

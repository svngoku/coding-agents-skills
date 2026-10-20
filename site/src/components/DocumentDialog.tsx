import { ExternalLink } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { MarkdownView } from "@/components/MarkdownView";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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

const DOCUMENT_GROUPS = [
  { label: "Skill", kind: "skill" },
  { label: "Reference guides", kind: "guide" },
  { label: "Scripts", kind: "script" },
] as const;

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
              <Select
                items={selection.skill.documents.map((document) => ({
                  value: document.path,
                  label: document.label,
                }))}
                value={selection.document.path}
                onValueChange={(value) => {
                  const next = selection.skill.documents.find((item) => item.path === value);
                  if (next) onSelect(next);
                }}
              >
                <SelectTrigger id="document-switcher" aria-label="Switch document" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent align="start">
                  {DOCUMENT_GROUPS.map((group) => {
                    const documents = selection.skill.documents.filter(
                      (item) => item.kind === group.kind,
                    );
                    if (!documents.length) return null;
                    return (
                      <SelectGroup key={group.kind}>
                        <SelectLabel>{group.label}</SelectLabel>
                        {documents.map((document) => (
                          <SelectItem key={document.path} value={document.path}>
                            {document.label}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>
          </DialogHeader>
          <ScrollArea className="min-h-0 flex-1">
            <div className="p-5">
              <MarkdownView html={selection.document.html} />
            </div>
          </ScrollArea>
        </DialogContent>
      ) : null}
    </Dialog>
  );
}

import {
  ArrowLeft,
  BookOpen,
  ExternalLink,
  FileText,
  Terminal,
} from "lucide-react";
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
import { Button, buttonVariants } from "@/components/ui/button";
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
  onBack: () => void;
};

const DOCUMENT_GROUPS = [
  { label: "Skill", kind: "skill" },
  { label: "Reference guides", kind: "guide" },
  { label: "Scripts", kind: "script" },
] as const;

export function DocumentDialog({
  selection,
  onClose,
  onSelect,
  onBack,
}: DocumentDialogProps) {
  return (
    <Dialog
      open={selection !== null}
      onOpenChange={(open) => (open ? undefined : onClose())}
    >
      {selection ? (
        <DialogContent className="flex h-[94dvh] w-[calc(100vw-1rem)] max-w-6xl flex-col gap-0 overflow-hidden rounded-2xl p-0 sm:max-w-6xl">
          <DialogHeader className="shrink-0 gap-2 border-b p-4 pr-12 sm:px-6 sm:py-5 sm:pr-12">
            <DialogDescription>
              <button
                type="button"
                onClick={onBack}
                className="inline-flex max-w-full items-center gap-1.5 text-xs hover:text-foreground"
                aria-label={`Back to ${selection.skill.title} details`}
              >
                <ArrowLeft className="size-3 shrink-0" aria-hidden="true" />
                <span className="min-w-0 break-all">
                  {selection.skill.name}
                </span>
              </button>
            </DialogDescription>
            <DialogTitle className="text-lg leading-snug font-medium tracking-tight [overflow-wrap:anywhere] sm:text-xl">
              {selection.document.label}
            </DialogTitle>
            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-2">
              <div className="w-full min-w-0 sm:max-w-sm">
                <Select
                  items={selection.skill.documents.map((document) => ({
                    value: document.path,
                    label: document.label,
                  }))}
                  value={selection.document.path}
                  onValueChange={(value) => {
                    const next = selection.skill.documents.find(
                      (item) => item.path === value,
                    );
                    if (next) onSelect(next);
                  }}
                >
                  <SelectTrigger
                    id="document-switcher"
                    aria-label="Switch document"
                    className="w-full bg-muted/30"
                  >
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
                            <SelectItem
                              key={document.path}
                              value={document.path}
                            >
                              {document.label}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      );
                    })}
                  </SelectContent>
                </Select>
              </div>
              <span className="text-xs tabular-nums text-muted-foreground">
                {new Intl.NumberFormat("en-US").format(
                  selection.document.lines,
                )}{" "}
                lines
              </span>
              <a
                href={selection.document.url}
                target="_blank"
                rel="noreferrer"
                className={cn(
                  buttonVariants({ variant: "ghost", size: "xs" }),
                  "gap-1.5 text-muted-foreground",
                )}
              >
                Source
                <ExternalLink className="size-3" aria-hidden="true" />
              </a>
            </div>
          </DialogHeader>
          <div className="flex min-h-0 flex-1">
            <aside className="hidden w-60 shrink-0 border-r bg-muted/25 lg:block">
              <ScrollArea className="h-full">
                <nav aria-label="Document navigation" className="space-y-5 p-3">
                  {DOCUMENT_GROUPS.map((group) => {
                    const documents = selection.skill.documents.filter(
                      (item) => item.kind === group.kind,
                    );
                    if (!documents.length) return null;
                    const Icon =
                      group.kind === "skill"
                        ? BookOpen
                        : group.kind === "guide"
                          ? FileText
                          : Terminal;
                    return (
                      <section key={group.kind} aria-label={group.label}>
                        <h3 className="mb-2 px-3 text-xs font-medium text-muted-foreground">
                          {group.label}
                        </h3>
                        <ul className="space-y-1">
                          {documents.map((document) => (
                            <li key={document.path}>
                              <Button
                                type="button"
                                variant="ghost"
                                onClick={() => onSelect(document)}
                                aria-current={
                                  document.path === selection.document.path
                                    ? "page"
                                    : undefined
                                }
                                className={cn(
                                  "h-auto w-full items-start justify-start gap-2 rounded-lg px-3 py-2.5 text-left text-xs leading-5 font-normal whitespace-normal",
                                  document.path === selection.document.path &&
                                    "bg-primary/8 font-medium text-primary hover:bg-primary/12 hover:text-primary",
                                )}
                              >
                                <Icon
                                  className="mt-0.5 size-3.5 shrink-0"
                                  aria-hidden="true"
                                />
                                <span className="min-w-0 [overflow-wrap:anywhere]">
                                  {document.label}
                                </span>
                              </Button>
                            </li>
                          ))}
                        </ul>
                      </section>
                    );
                  })}
                </nav>
              </ScrollArea>
            </aside>
            <ScrollArea
              key={`${selection.skill.name}/${selection.document.path}`}
              data-testid="document-scroll"
              className="min-h-0 min-w-0 flex-1"
            >
              <div className="mx-auto w-full max-w-[50rem] px-5 py-6 sm:px-9 sm:py-8">
                <p className="mb-6 font-mono text-[11px] leading-5 break-all text-muted-foreground">
                  skills/{selection.skill.name}/{selection.document.path}
                </p>
                <MarkdownView html={selection.document.html} />
              </div>
            </ScrollArea>
          </div>
        </DialogContent>
      ) : null}
    </Dialog>
  );
}

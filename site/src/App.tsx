import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { AlertTriangle, FolderGit2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "cn";
import type { DocumentSelection } from "@/components/DocumentDialog";
import { EmptyState } from "@/components/EmptyState";
import { FilterBar } from "@/components/FilterBar";
import { InstallCard } from "@/components/InstallCard";
import { MatrixTable } from "@/components/MatrixTable";
import { SkillCard } from "@/components/SkillCard";
import { StatsStrip } from "@/components/StatsStrip";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useGallery } from "@/lib/data";
import { applyTheme, readTheme, type Theme } from "@/lib/theme";
import type { GalleryDocument, Skill, SortKey, ViewMode } from "@/lib/types";

const CLI_URL = "https://github.com/vercel-labs/skills";

// The dialogs pull in the Markdown renderer and highlight.js, so load them on demand.
const SkillDetailsDialog = lazy(() =>
  import("@/components/SkillDetailsDialog").then((module) => ({
    default: module.SkillDetailsDialog,
  })),
);
const DocumentDialog = lazy(() =>
  import("@/components/DocumentDialog").then((module) => ({
    default: module.DocumentDialog,
  })),
);

export default function App() {
  const gallery = useGallery();
  const [theme, setTheme] = useState<Theme>(() => readTheme());
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<string | null>(null);
  const [details, setDetails] = useState<Skill | null>(null);
  const [documentSelection, setDocumentSelection] = useState<DocumentSelection | null>(null);
  const [sort, setSort] = useState<SortKey>("name");
  const [view, setView] = useState<ViewMode>("cards");

  const skills = gallery.data?.skills ?? [];

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return skills.filter((skill) => {
      if (category && skill.category !== category) return false;
      if (!needle) return true;
      return skill.haystack.includes(needle);
    });
  }, [skills, query, category]);

  const sorted = useMemo(() => {
    const list = [...visible];
    list.sort((a, b) => {
      switch (sort) {
        case "refs":
          return b.references.length - a.references.length || a.name.localeCompare(b.name);
        case "lines":
          return b.docsLines - a.docsLines || a.name.localeCompare(b.name);
        case "category":
          return a.category.localeCompare(b.category) || a.name.localeCompare(b.name);
        default:
          return a.name.localeCompare(b.name);
      }
    });
    return list;
  }, [visible, sort]);

  // Cards render after the payload arrives, so honour deep links once it is ready.
  useEffect(() => {
    if (gallery.status !== "ready") return;
    const hash = window.location.hash;
    if (!hash.startsWith("#skill-")) return;
    document.getElementById(hash.slice(1))?.scrollIntoView({ block: "start" });
  }, [gallery.status]);

  function toggleTheme() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    applyTheme(next);
    setTheme(next);
  }

  function readSkill(skill: Skill) {
    const skillDocument = skill.documents.find((item) => item.path === "SKILL.md");
    if (skillDocument) setDocumentSelection({ skill, document: skillDocument });
  }

  function openDocument(skill: Skill, item: GalleryDocument) {
    setDetails(null);
    setDocumentSelection({ skill, document: item });
  }

  function resetFilters() {
    setQuery("");
    setCategory(null);
  }

  const stats = gallery.data?.stats;

  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 focus:rounded-lg focus:bg-primary focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-primary-foreground"
      >
        Skip to content
      </a>

      <header className="border-b bg-card">
        <div className="mx-auto flex max-w-7xl flex-wrap items-start justify-between gap-4 px-4 py-6 sm:px-6 lg:px-8">
          <div className="min-w-0">
            <p className="text-xs font-medium text-primary">coding-agents-skills</p>
            <h1 className="mt-1 text-2xl font-bold text-balance sm:text-3xl">Skill gallery</h1>
            <p className="mt-1 max-w-2xl text-pretty text-sm text-muted-foreground">
              {stats ? stats.skills : "Fifteen"} production-ready skills for AI coding agents:
              design, testing, security, performance, deployment and AI frameworks, each with
              deep-dive reference guides and a skillgrade eval harness.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <a
              href={gallery.data?.repo.url ?? "https://github.com/svngoku/coding-agents-skills"}
              target="_blank"
              rel="noreferrer"
              className={cn(buttonVariants({ variant: "outline", size: "sm" }), "gap-1.5")}
            >
              <FolderGit2 className="size-4" aria-hidden="true" />
              Repository
            </a>
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
          </div>
        </div>
      </header>

      <main id="main" className="mx-auto w-full max-w-7xl flex-1 space-y-6 px-4 py-6 sm:px-6 lg:px-8">
        {gallery.status === "loading" ? <LoadingState /> : null}

        {gallery.status === "error" ? (
          <Alert variant="destructive" className="items-start">
            <AlertTriangle aria-hidden="true" />
            <AlertTitle>The gallery could not load</AlertTitle>
            <AlertDescription>
              {gallery.error}
              <div className="mt-3">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => window.location.reload()}
                >
                  Try again
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        ) : null}

        {gallery.status === "ready" && stats ? (
          <>
            <StatsStrip stats={stats} />
            <InstallCard slug={gallery.data.repo.slug} cliUrl={CLI_URL} />
            <FilterBar
              query={query}
              onQueryChange={setQuery}
              category={category}
              onCategoryChange={(value) => setCategory((current) => (current === value ? null : value))}
              categories={gallery.data.categories}
              sort={sort}
              onSortChange={setSort}
              view={view}
              onViewChange={setView}
              total={skills.length}
              visible={visible.length}
            />

            {!sorted.length ? (
              <EmptyState onReset={resetFilters} />
            ) : view === "cards" ? (
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {sorted.map((skill) => (
                  <SkillCard
                    key={skill.name}
                    skill={skill}
                    onDetails={setDetails}
                    onRead={readSkill}
                  />
                ))}
              </div>
            ) : (
              <MatrixTable skills={sorted} />
            )}
          </>
        ) : null}
      </main>

      <footer className="border-t py-6">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 text-sm text-muted-foreground sm:px-6 lg:px-8">
          <p>
            Built from <code className="font-mono text-xs">skills/</code>
            {stats
              ? ` — ${stats.documents} documents, ${new Intl.NumberFormat("en-US").format(stats.docsLines)} lines of Markdown.`
              : "."}
          </p>
          <div className="flex items-center gap-4">
            <a
              href={gallery.data?.repo.url ?? "https://github.com/svngoku/coding-agents-skills"}
              className="hover:text-foreground"
            >
              Repository
            </a>
            <a href="https://github.com/vercel-labs/skills" className="hover:text-foreground">
              Skills CLI
            </a>
          </div>
        </div>
      </footer>

      <Suspense fallback={null}>
        <SkillDetailsDialog
          key={details?.name ?? "closed"}
          skill={details}
          onClose={() => setDetails(null)}
          onRead={readSkill}
          onOpenDocument={openDocument}
        />
        <DocumentDialog
          selection={documentSelection}
          onClose={() => setDocumentSelection(null)}
          onSelect={(next) =>
            setDocumentSelection((current) =>
              current ? { skill: current.skill, document: next } : current,
            )
          }
        />
      </Suspense>
    </div>
  );
}

function LoadingState() {
  return (
    <div aria-busy="true" aria-live="polite" className="space-y-6">
      <span className="sr-only">Loading the skill gallery…</span>
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border bg-border sm:grid-cols-3 lg:grid-cols-5">
        {Array.from({ length: 5 }).map((_, index) => (
          <div key={index} className="space-y-2 bg-card px-4 py-3">
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-6 w-12" />
            <Skeleton className="h-3 w-24" />
          </div>
        ))}
      </div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {Array.from({ length: 6 }).map((_, index) => (
          <Card key={index} className="h-64 gap-3">
            <CardHeader className="space-y-2">
              <Skeleton className="h-5 w-2/3" />
              <Skeleton className="h-3 w-1/3" />
            </CardHeader>
            <CardContent className="space-y-2">
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-3 w-11/12" />
              <Skeleton className="h-3 w-4/5" />
              <Skeleton className="mt-4 h-6 w-2/3" />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

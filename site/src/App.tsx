import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowDown,
  ArrowUpRight,
  BookOpenCheck,
  FolderGit2,
  LibraryBig,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "cn";
import type { DocumentSelection } from "@/components/DocumentDialog";
import { CategoryNav } from "@/components/CategoryNav";
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
  const [documentSelection, setDocumentSelection] =
    useState<DocumentSelection | null>(null);
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
          return (
            b.references.length - a.references.length ||
            a.name.localeCompare(b.name)
          );
        case "lines":
          return b.docsLines - a.docsLines || a.name.localeCompare(b.name);
        case "category":
          return (
            a.category.localeCompare(b.category) || a.name.localeCompare(b.name)
          );
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

  function showDetails(skill: Skill) {
    setDetails(skill);
  }

  function readSkill(skill: Skill) {
    const skillDocument = skill.documents.find(
      (item) => item.path === "SKILL.md",
    );
    if (skillDocument) openDocument(skill, skillDocument);
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
  const repoUrl =
    gallery.data?.repo.url ?? "https://github.com/svngoku/coding-agents-skills";

  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 focus:rounded-lg focus:bg-primary focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-primary-foreground"
      >
        Skip to content
      </a>

      <header className="border-b bg-card/75">
        <div className="page-shell flex min-h-19 flex-wrap items-center justify-between gap-3 py-4">
          <a
            href="./"
            className="flex min-w-0 items-center gap-2.5"
            aria-label="Coding agents skills home"
          >
            <span className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-foreground text-background">
              <LibraryBig className="size-5" aria-hidden="true" />
            </span>
            <span className="font-heading text-[15px] font-medium tracking-tight">
              Agent skills
            </span>
            <span className="ml-1 hidden border-l pl-3 text-xs text-muted-foreground sm:inline">
              by svngoku
            </span>
          </a>
          <nav
            aria-label="Main navigation"
            className="flex items-center gap-2 sm:gap-5"
          >
            <a
              href="#library"
              className="hidden text-sm text-muted-foreground hover:text-foreground sm:inline"
            >
              Browse
            </a>
            <a
              href="#install"
              className="hidden text-sm text-muted-foreground hover:text-foreground sm:inline"
            >
              Get started
            </a>
            <a
              href={repoUrl}
              target="_blank"
              rel="noreferrer"
              aria-label="View the GitHub repository"
              className={cn(
                buttonVariants({ variant: "ghost", size: "sm" }),
                "gap-1.5 px-2 text-muted-foreground",
              )}
            >
              <FolderGit2 className="size-4" aria-hidden="true" />
              <span className="hidden min-[380px]:inline">GitHub</span>
            </a>
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
          </nav>
        </div>
      </header>

      <main id="main" tabIndex={-1} className="page-shell flex-1">
        <section aria-labelledby="intro-title" className="py-8 sm:py-10">
          <div className="grid items-center gap-8 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] lg:gap-16">
            <div className="max-w-xl">
              <h1 id="intro-title" className="intro-title">
                A working library for coding agents.
              </h1>
              <p className="mt-5 max-w-[46ch] text-[15px] leading-7 text-pretty text-muted-foreground sm:text-base">
                Focused skills, deep-dive guides, and evaluation harnesses. Find
                the context your next task needs, then bring it to your agent.
              </p>
              <div className="mt-7 flex flex-wrap items-center gap-4">
                <a
                  href="#library"
                  className={cn(
                    buttonVariants({ size: "lg" }),
                    "gap-2 rounded-lg px-5",
                  )}
                >
                  Explore the library
                  <ArrowDown className="size-4" aria-hidden="true" />
                </a>
                <span className="text-xs text-muted-foreground">
                  Open source. MIT licensed.
                </span>
              </div>
              {stats ? (
                <div className="mt-7">
                  <StatsStrip stats={stats} />
                </div>
              ) : null}
            </div>
            {gallery.status === "ready" ? (
              <InstallCard
                slug={gallery.data.repo.slug}
                cliUrl={CLI_URL}
                skills={skills}
              />
            ) : gallery.status === "error" ? (
              <Alert
                variant="destructive"
                className="items-start rounded-2xl bg-card p-5"
              >
                <AlertTriangle aria-hidden="true" />
                <AlertTitle>The gallery could not load</AlertTitle>
                <AlertDescription>
                  {gallery.error}
                  <div className="mt-3">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="text-foreground"
                      onClick={() => window.location.reload()}
                    >
                      Try again
                    </Button>
                  </div>
                </AlertDescription>
              </Alert>
            ) : (
              <div
                className="min-h-85 space-y-5 rounded-2xl border bg-card p-5"
                aria-hidden="true"
              >
                <Skeleton className="h-5 w-36" />
                <Skeleton className="h-9 w-full" />
                <div className="grid grid-cols-2 gap-3">
                  <Skeleton className="h-14" />
                  <Skeleton className="h-14" />
                </div>
                <Skeleton className="h-28 w-full" />
              </div>
            )}
          </div>
        </section>

        {gallery.status !== "error" ? (
          <section
            id="library"
            aria-labelledby="library-title"
            className="scroll-mt-6 border-t py-8"
          >
            <div className="mb-6">
              <h2
                id="library-title"
                className="text-2xl font-medium tracking-tight sm:text-[1.75rem]"
              >
                The skill library
              </h2>
            </div>

            {gallery.status === "loading" ? (
              <LoadingState />
            ) : (
              <div className="grid items-start gap-6 lg:grid-cols-[190px_minmax(0,1fr)] lg:gap-8">
                <aside className="hidden lg:block">
                  <div className="sticky top-6">
                    <CategoryNav
                      categories={gallery.data.categories}
                      category={category}
                      onChange={setCategory}
                      total={skills.length}
                    />
                    <div className="mx-3 mt-6 border-t pt-5">
                      <BookOpenCheck
                        className="mb-3 size-5 text-primary"
                        aria-hidden="true"
                      />
                      <p className="text-xs leading-5 text-muted-foreground">
                        More than a prompt. Inspect each skill&apos;s guides,
                        scripts, and eval files before installing.
                      </p>
                      <a
                        href={`${repoUrl}#contributing`}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-3 inline-flex items-center gap-1 text-xs font-medium hover:text-primary"
                      >
                        Contribute a skill
                        <ArrowUpRight className="size-3" aria-hidden="true" />
                      </a>
                    </div>
                  </div>
                </aside>
                <div className="min-w-0">
                  <FilterBar
                    query={query}
                    onQueryChange={setQuery}
                    category={category}
                    onCategoryChange={setCategory}
                    categories={gallery.data.categories}
                    sort={sort}
                    onSortChange={setSort}
                    view={view}
                    onViewChange={setView}
                    total={skills.length}
                    visible={visible.length}
                    onReset={resetFilters}
                  />
                  {!sorted.length ? (
                    <EmptyState onReset={resetFilters} />
                  ) : view === "cards" ? (
                    <div
                      data-testid="skill-grid"
                      className="grid items-stretch gap-4 sm:grid-cols-2 xl:grid-cols-3"
                    >
                      {sorted.map((skill) => (
                        <SkillCard
                          key={skill.name}
                          skill={skill}
                          onDetails={showDetails}
                          onRead={readSkill}
                        />
                      ))}
                    </div>
                  ) : (
                    <MatrixTable skills={sorted} onDetails={showDetails} />
                  )}
                </div>
              </div>
            )}
          </section>
        ) : null}
      </main>

      <footer className="mt-2 border-t py-7">
        <div className="page-shell flex flex-wrap items-center justify-between gap-5 text-xs text-muted-foreground">
          <div>
            <a
              href={repoUrl}
              className="inline-flex items-center gap-2 font-medium text-foreground"
            >
              <LibraryBig className="size-4" aria-hidden="true" />
              coding-agents-skills
            </a>
            <p className="mt-2 leading-6">
              {stats
                ? `${stats.documents} documents, ${stats.scripts} scripts, ${new Intl.NumberFormat("en-US").format(stats.docsLines)} documentation lines.`
                : "Skills and references, versioned on GitHub."}
            </p>
          </div>
          <nav
            aria-label="Footer navigation"
            className="flex flex-wrap items-center gap-5"
          >
            <a href={repoUrl} className="hover:text-foreground">
              Repository
            </a>
            <a href={CLI_URL} className="hover:text-foreground">
              Skills CLI
            </a>
            <a
              href={`${repoUrl}/blob/main/LICENSE`}
              className="hover:text-foreground"
            >
              MIT license
            </a>
          </nav>
        </div>
      </footer>

      <Suspense fallback={null}>
        {details ? (
          <SkillDetailsDialog
            key={details.name}
            skill={details}
            onClose={() => setDetails(null)}
            onRead={readSkill}
            onOpenDocument={openDocument}
          />
        ) : null}
        {documentSelection ? (
          <DocumentDialog
            selection={documentSelection}
            onClose={() => setDocumentSelection(null)}
            onBack={() => {
              if (documentSelection) {
                showDetails(documentSelection.skill);
                setDocumentSelection(null);
              }
            }}
            onSelect={(next) =>
              setDocumentSelection((current) =>
                current ? { skill: current.skill, document: next } : current,
              )
            }
          />
        ) : null}
      </Suspense>
    </div>
  );
}

function LoadingState() {
  return (
    <div
      aria-busy="true"
      aria-live="polite"
      className="grid gap-8 lg:grid-cols-[190px_minmax(0,1fr)]"
    >
      <span className="sr-only">Loading the skill gallery…</span>
      <div className="hidden space-y-3 lg:block" aria-hidden="true">
        {Array.from({ length: 8 }).map((_, index) => (
          <Skeleton key={index} className="h-10 w-full" />
        ))}
      </div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {Array.from({ length: 6 }).map((_, index) => (
          <Card
            key={index}
            className="min-h-90 gap-4 rounded-2xl border ring-0"
          >
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

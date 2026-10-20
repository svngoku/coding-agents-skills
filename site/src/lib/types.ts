export type Reference = {
  file: string;
  title: string;
  lines: number;
  url: string;
};

export type ScriptFile = {
  file: string;
  lines: number;
  url: string;
};

export type DocumentKind = "skill" | "guide" | "script";

export type GalleryDocument = {
  path: string;
  label: string;
  kind: DocumentKind;
  html: string;
  lines: number;
  url: string;
};

export type EvalInfo = {
  present: boolean;
  tasks: string[];
  threshold: number | null;
};

export type ArtifactCounts = {
  references: number;
  scripts: number;
  instructions: number;
  graders: number;
  rubrics: number;
  solutions: number;
  fixtures: number;
};

export type Skill = {
  name: string;
  title: string;
  category: string;
  description: string;
  trigger: string;
  capabilities: string[];
  useCases: string[];
  status: string;
  references: Reference[];
  scripts: ScriptFile[];
  documents: GalleryDocument[];
  eval: EvalInfo;
  artifacts: ArtifactCounts;
  docsLines: number;
  totalLines: number;
  haystack: string;
  skillUrl: string;
  dirUrl: string;
};

export type GalleryStats = {
  skills: number;
  references: number;
  scripts: number;
  evalTasks: number;
  docsLines: number;
  documents: number;
};

export type GalleryCategory = {
  name: string;
  count: number;
};

export type Gallery = {
  schemaVersion: number;
  generatedBy: string;
  repo: {
    url: string;
    slug: string;
    branch: string;
  };
  stats: GalleryStats;
  categories: GalleryCategory[];
  skills: Skill[];
};

export type SortKey = "name" | "refs" | "lines" | "category";

export type ViewMode = "cards" | "matrix";

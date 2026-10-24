import { useState } from "react";
import { ArrowUpRight, Check, Terminal } from "lucide-react";
import { CopyButton } from "@/components/CopyButton";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { Skill } from "@/lib/types";

type InstallCardProps = {
  slug: string;
  cliUrl: string;
  skills: Skill[];
};

const AGENTS = [
  { value: "claude-code", label: "Claude Code" },
  { value: "codex", label: "Codex" },
  { value: "opencode", label: "OpenCode" },
  { value: "cursor", label: "Cursor" },
];

type InstallMode = "skill" | "all" | "update";

export function InstallCard({ slug, cliUrl, skills }: InstallCardProps) {
  const [mode, setMode] = useState<InstallMode>("skill");
  const [agent, setAgent] = useState("claude-code");
  const [skill, setSkill] = useState(
    () =>
      skills.find((item) => item.name === "ddd")?.name ?? skills[0]?.name ?? "",
  );
  const [updateTarget, setUpdateTarget] = useState("all");
  const skillItems = skills.map((item) => ({
    value: item.name,
    label: item.title,
  }));
  const updateItems = [
    { value: "all", label: "All installed skills" },
    ...skillItems,
  ];
  const command =
    mode === "update"
      ? `npx skills update${updateTarget === "all" ? "" : ` ${updateTarget}`}`
      : `npx skills add ${slug} --skill ${mode === "all" ? "'*'" : skill} -a ${agent}`;

  return (
    <Card
      id="install"
      className="min-w-0 scroll-mt-6 gap-0 rounded-2xl border bg-card p-0 ring-0"
    >
      <CardHeader className="flex flex-row items-center justify-between gap-3 px-5 py-4">
        <h2 className="flex items-center gap-2 text-sm font-medium">
          <Terminal className="size-4 text-primary" aria-hidden="true" />
          Add to your agent
        </h2>
        <a
          href={cliUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          Skills CLI
          <ArrowUpRight className="size-3" aria-hidden="true" />
        </a>
      </CardHeader>
      <Separator />
      <CardContent className="space-y-4 p-5">
        <ToggleGroup
          value={[mode]}
          onValueChange={(next) => {
            const [value] = next;
            if (value) setMode(value as InstallMode);
          }}
          aria-label="Installation options"
          className="w-full gap-1 rounded-lg bg-muted p-1"
        >
          <ToggleGroupItem
            value="skill"
            className="min-w-0 flex-1 justify-center rounded-md px-2 text-xs aria-pressed:bg-card aria-pressed:shadow-xs"
          >
            One skill
          </ToggleGroupItem>
          <ToggleGroupItem
            value="all"
            className="min-w-0 flex-1 justify-center rounded-md px-2 text-xs aria-pressed:bg-card aria-pressed:shadow-xs"
          >
            All skills
          </ToggleGroupItem>
          <ToggleGroupItem
            value="update"
            className="min-w-0 flex-1 justify-center rounded-md px-2 text-xs aria-pressed:bg-card aria-pressed:shadow-xs"
          >
            Update
          </ToggleGroupItem>
        </ToggleGroup>

        <div className="grid min-h-17 grid-cols-2 gap-3">
          {mode === "update" ? (
            <div className="col-span-2 space-y-1.5">
              <label
                htmlFor="update-target"
                className="text-xs text-muted-foreground"
              >
                Skills to update
              </label>
              <Select
                items={updateItems}
                value={updateTarget}
                onValueChange={(value) => {
                  if (value) setUpdateTarget(value);
                }}
              >
                <SelectTrigger id="update-target" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent align="start">
                  {updateItems.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ) : (
            <>
              <div
                className={
                  mode === "all"
                    ? "col-span-2 space-y-1.5"
                    : "min-w-0 space-y-1.5"
                }
              >
                <label
                  htmlFor="install-agent"
                  className="text-xs text-muted-foreground"
                >
                  Your agent
                </label>
                <Select
                  items={AGENTS}
                  value={agent}
                  onValueChange={(value) => {
                    if (value) setAgent(value);
                  }}
                >
                  <SelectTrigger id="install-agent" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent align="start">
                    {AGENTS.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {mode === "skill" ? (
                <div className="min-w-0 space-y-1.5">
                  <label
                    htmlFor="install-skill"
                    className="text-xs text-muted-foreground"
                  >
                    Skill
                  </label>
                  <Select
                    items={skillItems}
                    value={skill}
                    onValueChange={(value) => {
                      if (value) setSkill(value);
                    }}
                  >
                    <SelectTrigger id="install-skill" className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent align="start">
                      {skillItems.map((item) => (
                        <SelectItem key={item.value} value={item.value}>
                          {item.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ) : null}
            </>
          )}
        </div>

        <div className="overflow-hidden rounded-xl bg-[#202823] text-[#f1f5ef]">
          <div className="flex items-center justify-between border-b border-white/10 px-4 py-1.5">
            <span className="text-[11px] text-[#b7c6ba]">
              Run in your terminal
            </span>
            <CopyButton
              value={command}
              label="Installation command"
              className="text-[#f1f5ef] hover:bg-white/10 hover:text-white focus-visible:ring-white/50"
            />
          </div>
          <pre
            tabIndex={0}
            role="region"
            aria-label="Installation command"
            className="min-h-22 px-4 py-4 font-mono text-[12px] leading-6 whitespace-pre-wrap [overflow-wrap:anywhere]"
          >
            <span aria-hidden="true" className="mr-2 text-[#a9d1b5]">
              $
            </span>
            <code>{command}</code>
          </pre>
        </div>
        <p className="flex items-start gap-1.5 text-[11px] leading-5 text-muted-foreground">
          <Check
            className="mt-1 size-3 shrink-0 text-primary"
            aria-hidden="true"
          />
          Requires Node.js 22.20 or later. Also supports other agents through
          the CLI.
        </p>
      </CardContent>
    </Card>
  );
}

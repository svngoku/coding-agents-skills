import { ArrowUpRight, Terminal } from "lucide-react";
import { CopyButton } from "@/components/CopyButton";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "cn";

type InstallCardProps = {
  slug: string;
  cliUrl: string;
};

type Command = {
  label: string;
  value: string;
};

export function InstallCard({ slug, cliUrl }: InstallCardProps) {
  const commands: Command[] = [
    { label: "One skill", value: `npx skills add ${slug} --skill ddd -a claude-code` },
    { label: "Every skill", value: `npx skills add ${slug} --skill '*' -a claude-code` },
    { label: "Update all", value: "npx skills update" },
    { label: "Update one", value: "npx skills update ddd" },
  ];

  return (
    <Card className="gap-4">
      <CardHeader className="gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <CardTitle className="flex items-center gap-2 text-base">
            <Terminal className="size-4 text-primary" aria-hidden="true" />
            Install and update the skills
          </CardTitle>
          <CardDescription className="mt-1 max-w-2xl text-pretty">
            The{" "}
            <a
              href={cliUrl}
              className="font-medium text-primary underline-offset-4 hover:underline"
              target="_blank"
              rel="noreferrer"
            >
              open agent skills CLI
            </a>{" "}
            installs them into OpenCode, Claude Code, Codex, Cursor and 75 more agents. Requires
            Node.js 22.20 or later. Swap <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]">ddd</code>{" "}
            for any skill name.
          </CardDescription>
        </div>
        <a
          href={cliUrl}
          target="_blank"
          rel="noreferrer"
          className={cn(buttonVariants({ variant: "outline", size: "sm" }), "shrink-0 gap-1.5")}
        >
          CLI documentation
          <ArrowUpRight className="size-3.5" aria-hidden="true" />
        </a>
      </CardHeader>
      <CardContent className="grid gap-3 sm:grid-cols-2">
        {commands.map((command) => (
          <div key={command.label} className="min-w-0 rounded-lg border bg-muted/40 p-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-foreground">{command.label}</span>
              <CopyButton value={command.value} label={command.label} />
            </div>
            <pre
              tabIndex={0}
              className="mt-1 overflow-x-auto font-mono text-xs leading-5 text-muted-foreground"
            >
              <code>{command.value}</code>
            </pre>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

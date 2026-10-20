import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Skill } from "@/lib/types";

const KINDS = [
  { key: "references", label: "Refs" },
  { key: "scripts", label: "Scripts" },
  { key: "instructions", label: "Tasks" },
  { key: "graders", label: "Graders" },
  { key: "rubrics", label: "Rubrics" },
  { key: "solutions", label: "Solutions" },
  { key: "fixtures", label: "Fixtures" },
] as const;

type MatrixTableProps = {
  skills: Skill[];
};

export function MatrixTable({ skills }: MatrixTableProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Artifact matrix</CardTitle>
        <CardDescription>
          What each skill ships: reference guides, runnable scripts and a skillgrade eval harness.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Artifact matrix">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-4">Skill</TableHead>
                {KINDS.map((kind) => (
                  <TableHead key={kind.key} className="text-center">
                    {kind.label}
                  </TableHead>
                ))}
                <TableHead className="pr-4 text-right">Doc lines</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {skills.map((skill) => (
                <TableRow key={skill.name} data-category={skill.category}>
                  <TableCell className="pl-4">
                    <a href={`#skill-${skill.name}`} className="font-medium hover:text-primary">
                      {skill.title}
                    </a>
                    <span className="ml-2 font-mono text-xs text-muted-foreground">
                      {skill.name}
                    </span>
                  </TableCell>
                  {KINDS.map((kind) => {
                    const count = skill.artifacts[kind.key];
                    return (
                      <TableCell
                        key={kind.key}
                        className={
                          count
                            ? "text-center font-medium tabular-nums text-primary"
                            : "text-center text-muted-foreground/50"
                        }
                      >
                        {count || "·"}
                      </TableCell>
                    );
                  })}
                  <TableCell className="pr-4 text-right tabular-nums text-muted-foreground">
                    {skill.docsLines}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
  { key: "references", label: "Guides" },
  { key: "scripts", label: "Scripts" },
  { key: "instructions", label: "Tasks" },
  { key: "graders", label: "Graders" },
  { key: "rubrics", label: "Rubrics" },
  { key: "solutions", label: "Solutions" },
  { key: "fixtures", label: "Fixtures" },
] as const;

type MatrixTableProps = {
  skills: Skill[];
  onDetails: (skill: Skill) => void;
};

export function MatrixTable({ skills, onDetails }: MatrixTableProps) {
  return (
    <Card className="gap-0 rounded-2xl border p-0 ring-0">
      <CardHeader className="gap-1.5 border-b p-5">
        <CardTitle>
          <h3 className="text-base">Compare what ships</h3>
        </CardTitle>
        <CardDescription className="text-xs leading-5">
          Guides, scripts, and evaluation files for every skill. Select a skill
          to inspect it.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        <Table
          className="min-w-192 text-xs"
          containerProps={{
            tabIndex: 0,
            role: "region",
            "aria-label": "Artifact matrix",
          }}
        >
          <caption className="sr-only">
            Artifact counts and documentation size for the filtered skills.
          </caption>
          <TableHeader className="bg-muted/40">
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
                  <button
                    type="button"
                    onClick={() => onDetails(skill)}
                    className="block max-w-60 py-2 text-left text-sm font-medium whitespace-normal hover:text-primary"
                  >
                    {skill.title}
                  </button>
                  <span className="mb-2 block font-mono text-[11px] text-muted-foreground">
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
                          ? "text-center font-medium tabular-nums"
                          : "text-center tabular-nums text-muted-foreground"
                      }
                    >
                      {count}
                    </TableCell>
                  );
                })}
                <TableCell className="pr-4 text-right tabular-nums text-muted-foreground">
                  {new Intl.NumberFormat("en-US").format(skill.docsLines)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

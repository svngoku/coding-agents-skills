#!/usr/bin/env node
// Golden-principles linter. Zero dependencies. Exit 1 on any violation.
//   GP-1 AGENTS.md is a map (<150 lines)       GP-2 Required docs exist
//   GP-3 Diagrams are Mermaid only             GP-4 No secrets committed
//   GP-5 Active plans have frontmatter         GP-6 Relative doc links resolve
//   GP-7 CI is hardened (permissions, pinned refs)
const fs = require("fs");
const path = require("path");

const errors = [];
const fail = (gp, file, msg) => errors.push(`${gp} ${file}: ${msg}`);
const read = (f) => fs.readFileSync(f, "utf8");
const exists = (f) => fs.existsSync(f);
const SKIP = new Set(["node_modules", ".git", "dist", "build", ".venv", "vendor", ".next", "coverage", "__pycache__"]);

function walk(dir, out = []) {
  if (!exists(dir)) return out;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(e.name)) continue;
    const p = path.join(dir, e.name);
    e.isDirectory() ? walk(p, out) : out.push(p);
  }
  return out;
}

const all = walk(".");
const mdFiles = all.filter((f) => f.endsWith(".md"));

// GP-1
if (!exists("AGENTS.md")) fail("GP-1", "AGENTS.md", "missing");
else if (read("AGENTS.md").split("\n").length >= 150)
  fail("GP-1", "AGENTS.md", "must stay under 150 lines (map, not manual)");

// GP-2
for (const d of ["ARCHITECTURE", "DESIGN", "PLANS", "QUALITY_SCORE", "RELIABILITY", "SECURITY", "PRODUCT_SENSE"])
  if (!exists(`docs/${d}.md`)) fail("GP-2", `docs/${d}.md`, "missing");

// GP-3: no ASCII-art diagrams outside mermaid fences; no empty mermaid; no other diagram formats
const BOX = /[┌┐└┘├┤┬┴┼│─╭╮╯╰]|\+-{3,}\+/;
for (const f of mdFiles) {
  let fence = null;
  const src = read(f);
  src.split("\n").forEach((line, i) => {
    const m = line.match(/^\s*(```+|~~~+)\s*(\w*)/);
    if (m) { fence = fence ? null : m[2] || "plain"; return; }
    if ((fence === null || fence === "plain" || fence === "text") && BOX.test(line))
      fail("GP-3", `${f}:${i + 1}`, "ASCII-art diagram, use a Mermaid block instead");
  });
  for (const b of src.matchAll(/^```mermaid\n([\s\S]*?)^```/gm))
    if (!b[1].trim()) fail("GP-3", f, "empty mermaid block");
}
for (const f of all) if (/\.(puml|plantuml|dot|gv)$/.test(f))
  fail("GP-3", f, "non-Mermaid diagram source, convert to .mmd");

// GP-4: .env ignored + secret-shaped strings
if (exists(".env")) {
  const gi = exists(".gitignore") ? read(".gitignore") : "";
  if (!/^\.env$/m.test(gi)) fail("GP-4", ".env", "present but not in .gitignore");
}
const SECRETS = [
  [/AKIA[0-9A-Z]{16}/, "AWS access key id"],
  [/-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----/, "private key block"],
  [/gh[pousr]_[A-Za-z0-9]{36,}/, "GitHub token"],
  [/xox[baprs]-[A-Za-z0-9-]{10,}/, "Slack token"],
  [/sk-[A-Za-z0-9_-]{32,}/, "API secret key (sk-...)"],
  [/AIza[0-9A-Za-z_-]{35}/, "Google API key"],
];
const BINARY = /\.(png|jpe?g|gif|webp|ico|pdf|zip|gz|woff2?|ttf|lock)$/i;
for (const f of all) {
  if (BINARY.test(f) || f.endsWith("golden_principles.js")) continue;
  let text; try { text = read(f); } catch { continue; }
  for (const [re, label] of SECRETS)
    if (re.test(text)) fail("GP-4", f, `looks like a committed secret (${label})`);
}

// GP-5
for (const f of walk("docs/exec-plans/active").filter((x) => x.endsWith(".md")))
  if (!/^---\n[\s\S]*?\bstatus:/.test(read(f))) fail("GP-5", f, "missing frontmatter with status");

// GP-6
for (const f of mdFiles) {
  let fence = false;
  read(f).split("\n").forEach((line) => {
    if (/^\s*(```|~~~)/.test(line)) { fence = !fence; return; }
    if (fence) return;
    for (const m of line.matchAll(/\]\((?!https?:|mailto:|#)([^)#\s]+)/g)) {
      const target = path.resolve(path.dirname(f), m[1]);
      if (!exists(target)) fail("GP-6", f, `broken link -> ${m[1]}`);
    }
  });
}

// GP-7: workflows declare permissions and never use floating branch refs
for (const f of walk(".github/workflows").filter((x) => /\.ya?ml$/.test(x))) {
  const y = read(f);
  if (!/^permissions:/m.test(y)) fail("GP-7", f, "missing top-level `permissions:` (least privilege)");
  for (const m of y.matchAll(/uses:\s*([^\s#]+)@(main|master|latest)\b/g))
    fail("GP-7", f, `action ${m[1]} pinned to floating ref @${m[2]}`);
}

if (errors.length) {
  console.error(errors.join("\n") + `\n\n${errors.length} golden-principle violation(s)`);
  process.exit(1);
}
console.log("golden principles: all clear");

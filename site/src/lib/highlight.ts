import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import http from "highlight.js/lib/languages/http";
import ini from "highlight.js/lib/languages/ini";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import nginx from "highlight.js/lib/languages/nginx";
import plaintext from "highlight.js/lib/languages/plaintext";
import python from "highlight.js/lib/languages/python";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

const LANGUAGES = {
  bash,
  http,
  ini,
  javascript,
  json,
  markdown,
  nginx,
  plaintext,
  python,
  sql,
  typescript,
  xml,
  yaml,
};

for (const [name, definition] of Object.entries(LANGUAGES)) {
  hljs.registerLanguage(name, definition);
}

/** Fence aliases used across the skills; everything unknown renders as plain text. */
const ALIASES: Record<string, string> = {
  sh: "bash",
  shell: "bash",
  zsh: "bash",
  console: "bash",
  js: "javascript",
  jsx: "javascript",
  ts: "typescript",
  tsx: "typescript",
  html: "xml",
  hcl: "ini",
  tf: "ini",
  terraform: "ini",
  yml: "yaml",
  toml: "ini",
  text: "plaintext",
  txt: "plaintext",
  prisma: "plaintext",
  plain: "plaintext",
};

/**
 * Highlight every code block inside `root` exactly once. The Markdown HTML is
 * produced by scripts/site_markdown.py, so this only adds token spans.
 */
export function highlightWithin(root: HTMLElement): void {
  const blocks = root.querySelectorAll<HTMLElement>("pre code");
  blocks.forEach((block) => {
    if (block.classList.contains("language-mermaid")) return;
    if (block.dataset.highlighted === "yes") return;

    const declared = Array.from(block.classList)
      .find((className) => className.startsWith("language-"))
      ?.slice("language-".length);
    const language = declared ? (ALIASES[declared] ?? declared) : "";
    const resolved =
      language && hljs.getLanguage(language) ? language : "plaintext";

    if (declared) block.classList.remove(`language-${declared}`);
    block.classList.add(`language-${resolved}`);
    hljs.highlightElement(block);
  });
}

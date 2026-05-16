# coding-agents-skills — Agent Instructions

This document contains instructions for AI coding agents managing this repository.

## Repository Overview

This is a **curated collection of specialized skills for AI coding agents**. Each skill lives in `skills/<skill-name>/` and is designed to be loaded by agents to enhance their capabilities in specific domains (frameworks, deployment platforms, design patterns, etc.).

The repo is **documentation-first**: the primary artifact is Markdown. There is no build system, no tests, no dependencies. Quality is ensured through consistent structure and peer review.

## Repository Structure

```
coding-agents-skills/
├── skills/
│   ├── <skill-name>/          # One directory per skill
│   │   ├── SKILL.md           # REQUIRED — core skill definition
│   │   ├── references/        # OPTIONAL — deep-dive docs
│   │   └── AGENTS.md          # OPTIONAL — skill-specific agent notes
│   └── ...
├── README.md                  # REQUIRED — human-facing overview
└── AGENTS.md                  # THIS FILE
```

## Skill Conventions

Every skill MUST follow these conventions. Inconsistencies break the loading contract for agent frameworks.

### 1. YAML Frontmatter (REQUIRED)

Every `SKILL.md` MUST begin with YAML frontmatter in this exact format:

```yaml
---
name: <kebab-case-skill-name>
description: >
  <trigger sentence>. Use this skill whenever the user mentions <keywords>,
  wants to <actions>, or works with <tools/frameworks>. Also trigger for
  <related terms>.
---
```

Rules for `description`:
- Start with a **trigger sentence** — when/how the skill should be activated
- Mention **keywords, tools, CLI commands, API names** — these are how agents match
- Include **"Also trigger for"** with related terms users might say
- Keep it under 500 characters when possible; never exceed 1000
- Use `>` folded style for multi-line descriptions

### 2. SKILL.md Structure

After the frontmatter, use this structure:

```markdown
# <Human-Readable Skill Name>

[Brief 1-paragraph overview — what problem this skill solves]

## Quick Reference
[Table linking to reference files — ONLY if references/ exists]

## Core Workflow / Usage
[Step-by-step instructions for the most common tasks]

## <Topic Sections>
[Concrete examples, code snippets, configuration samples]

## Anti-Patterns to Avoid
[Common mistakes specific to this domain]

## When to Use / Not Use
[Decision criteria — saves tokens by avoiding irrelevant skill loads]
```

Guidelines:
- Use `##` (H2) for major sections, `###` (H3) for subsections
- Code blocks MUST have language tags (e.g. ` ```python `)
- Tables are preferred for comparisons and option references
- Keep the main `SKILL.md` focused — deep details go in `references/`
- NEVER include installation instructions for the skill loader itself (that's the consuming agent's concern)

### 3. references/ Directory (OPTIONAL)

Create `references/` only when the skill has deep-dive topics that would bloat `SKILL.md`:

```
references/
├── <topic>.md          # One file per deep topic
└── ...
```

Reference file conventions:
- Named in `kebab-case.md`
- Linked from `SKILL.md` Quick Reference table
- Self-contained — an agent should be able to read just this file and understand the topic
- Include code examples specific to the reference topic

### 4. Naming Conventions

| Element | Format | Example |
|---------|--------|---------|
| Skill directory | `kebab-case` | `smolagents`, `unsloth-hf-jobs` |
| `name` in frontmatter | `kebab-case` | `smolagents`, `unsloth-hf-jobs` |
| Reference files | `kebab-case.md` | `api-reference.md` |
| Section headers | Title Case | `## Core Workflow` |

## Adding a New Skill

Follow this workflow precisely when adding a new skill:

### Step 1: Create the directory structure

```bash
mkdir -p skills/<skill-name>/references  # references/ is optional
```

### Step 2: Write SKILL.md

Use the template from "Skill Conventions" above. Ensure:
- [ ] YAML frontmatter with `name` and `description`
- [ ] Clear trigger conditions in description
- [ ] Practical code examples with language tags
- [ ] "Anti-Patterns" section
- [ ] "When to Use / Not Use" section

### Step 3: Add reference files (if needed)

Only if `SKILL.md` would exceed ~300 lines without them.

### Step 4: Update README.md

The root `README.md` MUST be updated whenever skills change:

- [ ] Add the skill to the **Project Structure** tree
- [ ] Add a numbered section under **Available Skills**
- [ ] Mark status as `✅ Complete` (or `🚧 In Progress` if incomplete)
- [ ] Include: Description, Capabilities (bullet list), Use Cases (bullet list)
- [ ] If the skill has references, note the count: `Reference Files: N guides`
- [ ] Update the **Roadmap** checklist — mark `[x]` for completed skills

### Step 5: Self-review checklist

Before finishing, verify:
- [ ] All links in `SKILL.md` are relative (e.g. `references/topic.md`)
- [ ] Code blocks have language tags
- [ ] No broken Markdown tables
- [ ] YAML frontmatter is valid (no unescaped colons in description if not quoted)
- [ ] Description is under 1000 characters
- [ ] Skill name in frontmatter matches directory name

## Updating an Existing Skill

When modifying an existing skill:

1. **Read the current `SKILL.md`** first — understand the existing voice and structure
2. **Make minimal changes** — don't restructure unless necessary
3. **Preserve the frontmatter** — only update description if the skill's scope changes
4. **Update reference files** if adding deep-dive content
5. **Update README.md** if:
   - The skill's capabilities change significantly
   - Reference files are added/removed
   - Status changes (e.g., from In Progress to Complete)

## README Maintenance Rules

The `README.md` is the human-facing entry point. Keep it accurate:

- **Project Structure tree** must match actual `ls skills/`
- **Skill count** in "Available Skills" must match actual directories
- **Roadmap** must reflect reality — don't claim things are done that aren't
- **Related Projects** links should work (spot-check periodically)
- **Status badges** (`✅ Complete`, `🚧 In Progress`, `⏳ Planned`) must be honest

When updating README after adding a skill:
- Insert the skill in the tree in **alphabetical order** OR at the end of its category
- Number "Available Skills" sections sequentially
- Add `---` horizontal rules between skill sections for visual separation

## Common Mistakes to Avoid

- **Missing YAML frontmatter** — This breaks agent loading in frameworks like genai-tk
- **Overlong SKILL.md** — If it exceeds 400 lines, consider splitting into references/
- **Vague descriptions** — "Use this for Python stuff" is not a trigger. Be specific: tools, commands, API names
- **Forgetting README updates** — The README and skills/ must stay in sync
- **Using absolute file paths** in links — Always relative: `references/topic.md`
- **Missing anti-patterns** — These save agents from common mistakes; always include them
- **Inconsistent naming** — Directory `foo-bar` but frontmatter `name: foo_bar`
- **No "When NOT to Use"** — Agents waste tokens loading irrelevant skills

## Git Workflow

This repo uses simple trunk-based development:

1. **Main branch is `main`**
2. **No required PRs for small fixes** — edit directly if confidence is high
3. **For new skills or major rewrites** — create a branch, commit, and summarize changes
4. **Commit message style** — Imperative mood, descriptive:
   - `Add unsloth-hf-jobs skill for GPU fine-tuning`
   - `Update Scalingo skill with SecNumCloud region details`
   - `Fix broken links in DDD references`
5. **Do NOT commit** unless explicitly asked by the user

## File Editing Rules

When editing files in this repo:

- **Prefer `edit` over `write`** for existing files — preserves structure
- **Read before editing** — Always read the file first to understand context
- **Batch reads** — When updating README, read all skill SKILL.md files in parallel to verify content
- **Preserve formatting** — Match existing indentation, heading styles, table formats
- **No trailing whitespace** — Clean up if you notice it

## Markdown Quality Standards

- All headings follow ATX style (`# ` not `===` underlines)
- Code blocks fenced with triple backticks
- Tables use standard GFM pipe syntax with header separator
- Lists use `-` for bullets, `1.` for numbered
- No HTML in Markdown except for `<details>` / `<summary>` when necessary
- Line length: prefer under 120 chars, but don't hard-wrap prose

## No Build, No Test, No Deploy

This repository has:
- **No build system** — Markdown is the artifact
- **No tests** — Quality ensured through structure and review
- **No CI/CD** — Manual commits
- **No dependencies** — Pure documentation

Treat it as a **documentation codebase**. The "tests" are:
1. Does `SKILL.md` load correctly in an agent framework?
2. Are all relative links valid?
3. Is YAML frontmatter parseable?
4. Does README match the actual directory structure?

## When in Doubt

- Read 2–3 existing `SKILL.md` files to understand the house style
- The `ddd/` and `smolagents/` skills are the most mature — use them as reference
- Keep it simple: one skill, one domain, clear triggers, practical examples

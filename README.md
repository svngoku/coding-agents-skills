# Coding Agents Skills

> A curated collection of specialized skills for AI coding agents, designed to enhance agent capabilities across software development, architecture, testing, security, performance, deployment, and AI framework integration.

## 📖 Overview

This repository provides production-ready skill modules for AI coding agents, enabling them to:
- Apply software design and architecture patterns (DDD, microservices, database design)
- Design and review APIs with production-grade conventions
- Write and maintain automated tests across the full test pyramid
- Harden applications against OWASP-class vulnerabilities
- Find and fix performance problems by measuring first
- Work with modern AI frameworks and tools (LangChain, SmolAgents, genai-tk)
- Post-train and fine-tune LLMs (prime-rl, Unsloth)
- Deploy and operate applications on cloud platforms (Scalingo)
- Follow best practices in code generation, refactoring, and UI development
- Integrate seamlessly with development workflows

Each skill is self-contained with comprehensive documentation, reference materials, and practical examples.

## 📦 Installing the Skills

The collection installs with the [open agent skills CLI](https://github.com/vercel-labs/skills)
(`npx skills`), which supports OpenCode, Claude Code, Codex, Cursor and
[75 more agents](https://github.com/vercel-labs/skills/blob/main/docs/agents.md). Requires Node.js
22.20 or later.

```bash
# Browse what this repository ships, without installing anything
npx skills add svngoku/coding-agents-skills --list

# Interactive: choose the skills and the agents to install them for
npx skills add svngoku/coding-agents-skills
```

### Install one skill

```bash
# Just DDD, for Claude Code
npx skills add svngoku/coding-agents-skills --skill ddd -a claude-code

# Several skills at once
npx skills add svngoku/coding-agents-skills --skill ddd --skill langchain -a claude-code

# Or point straight at the skill's folder in this repository
npx skills add https://github.com/svngoku/coding-agents-skills/tree/main/skills/ddd
```

### Install every skill

```bash
# All 15 skills, for one agent
npx skills add svngoku/coding-agents-skills --skill '*' -a claude-code

# All skills, for every agent the CLI can install to, without prompts
npx skills add svngoku/coding-agents-skills --all
```

In project scope, agents whose project directory does not exist yet are skipped; `-g` installs at
user level instead.

### Where they land

| Scope             | Flag | Location                                                         |
| ----------------- | ---- | ---------------------------------------------------------------- |
| Project (default) | —    | The agent's project skills directory, e.g. `.claude/skills/ddd`   |
| Global            | `-g` | The agent's user-level skills directory, e.g. `~/.claude/skills/ddd` |

Directories differ per agent and scope: Claude Code uses `.claude/skills/` (or `~/.claude/skills/`
when global), while Cursor, Codex and Copilot share `.agents/skills/` in a project — see the
[agent table](https://github.com/vercel-labs/skills/blob/main/docs/agents.md). `--copy` installs
plain file copies instead of linked entries. Add `-y` to skip prompts in scripts and CI:

```bash
npx skills add svngoku/coding-agents-skills --skill ddd -g -a claude-code -y
```

### Updating the skills

Project installs record their source in `skills-lock.json` and global installs (`-g`) record
theirs in `~/skills-lock.json`; `update` re-reads the file for the scope it runs against.

```bash
# Update every skill in the current project
npx skills update

# Update one skill, or several
npx skills update ddd
npx skills update ddd langchain

# Restrict the run to one scope, or skip the scope prompt
npx skills update -g
npx skills update -p
npx skills update -y
```

| Option          | Description                                                        |
| --------------- | ------------------------------------------------------------------ |
| `-g, --global`  | Only update global skills                                          |
| `-p, --project` | Only update project skills                                         |
| `-y, --yes`     | Skip the scope prompt: project inside a project, global otherwise  |
| `<skills...>`   | Update the named skills instead of every installed skill            |

Check what is installed with `npx skills list`, and uninstall with `npx skills remove ddd`.

## 🗂️ Project Structure

```text
coding-agents-skills/
├── skills/
│   ├── adaption-ai/                    # Adaption AI SDK for synthetic data augmentation
│   │   ├── SKILL.md
│   │   ├── references/
│   │   │   ├── api-reference.md
│   │   │   └── guides.md
│   │   └── scripts/
│   │       ├── async_pipelines.py
│   │       └── e2e_pipeline.py
│   │
│   ├── api-design/                     # REST/HTTP API design & review
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── error-handling.md
│   │       ├── pagination-filtering.md
│   │       └── versioning-evolution.md
│   │
│   ├── database-design/                # Relational & NoSQL schema design
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── indexing-and-query-tuning.md
│   │       ├── migrations-zero-downtime.md
│   │       └── normalization-and-keys.md
│   │
│   ├── ddd/                            # Domain-Driven Design
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── strategic-design.md
│   │       ├── tactical-design.md
│   │       ├── architecture-patterns.md
│   │       ├── event-storming.md
│   │       ├── python-patterns.md
│   │       ├── typescript-patterns.md
│   │       └── code-review.md
│   │
│   ├── genai-tk/                       # GenAI Toolkit (YAML-driven agent framework)
│   │   ├── SKILL.md
│   │   ├── genai-tk-skill.skill        # Packaged skill archive (zip)
│   │   └── references/
│   │       ├── agents.md
│   │       ├── baml-structured.md
│   │       ├── cli-and-init.md
│   │       ├── configuration.md
│   │       ├── models.md
│   │       ├── rag.md
│   │       └── workflows.md
│   │
│   ├── langchain/                      # LangChain framework
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── deep-agents.md
│   │       ├── langgraph.md
│   │       ├── middleware.md
│   │       ├── multi-agent.md
│   │       └── retrieval.md
│   │
│   ├── microservices-patterns/         # Distributed systems patterns
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── observability-tracing.md
│   │       ├── resilience-patterns.md
│   │       └── saga-outbox.md
│   │
│   ├── performance-optimization/       # Profiling & optimization
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── backend-optimization.md
│   │       ├── frontend-performance.md
│   │       └── profiling-tools.md
│   │
│   ├── prime-rl/                       # PrimeIntellect prime-rl post-training (RL, SFT, distillation)
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── algorithms.md
│   │       ├── configs.md
│   │       ├── eval.md
│   │       ├── install.md
│   │       ├── monitoring.md
│   │       ├── scaling.md
│   │       └── training-runs.md
│   │
│   ├── scalingo/                       # Scalingo European PaaS deployment
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── addons-databases.md
│   │       ├── buildpacks.md
│   │       ├── cli-reference.md
│   │       ├── deployment.md
│   │       ├── manifest-review-apps.md
│   │       ├── scaling-autoscaler.md
│   │       ├── terraform-iac.md
│   │       └── tutorials.md
│   │
│   ├── security-best-practices/        # Application security hardening
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── jwt-oauth.md
│   │       └── threat-modeling.md
│   │
│   ├── smolagents/                     # Hugging Face SmolAgents
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── models.md
│   │       ├── patterns.md
│   │       └── tools.md
│   │
│   ├── testing-patterns/               # Automated testing strategies
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── flaky-tests-ci.md
│   │       ├── js-ts-testing.md
│   │       └── python-testing.md
│   │
│   ├── ui/                             # UI/UX best practices for agent-built interfaces
│   │   └── SKILL.md
│   │
│   └── unsloth-hf-jobs/                # Unsloth fine-tuning on Hugging Face Jobs
│       ├── SKILL.md
│       └── scripts/
│           ├── continued-pretraining.py
│           ├── sft-gemma3-vlm.py
│           └── sft-qwen3-vl.py
│
├── README.md
├── AGENTS.md
├── LICENSE
├── docs/
│   └── eval-guidelines.md           # skillgrade eval conventions & coverage
├── scripts/
│   ├── run-evals.sh                 # local eval runner (no CI)
│   ├── build-site.py                # extracts skills/ into site/public/gallery.json
│   └── site_markdown.py             # stdlib Markdown renderer used by the generator
├── site/                            # Vite + React + shadcn/ui gallery app
│   ├── src/                         # cards, dialogs, filters, theme, fonts
│   ├── public/gallery.json          # generated payload (committed)
│   ├── package.json
│   └── vite.config.ts
├── .github/workflows/
│   └── pages.yml                    # builds the app and deploys it to Pages
└── .gitignore                       # ignores .evals/, node_modules and dist
```

Every skill also ships a skillgrade evaluation harness alongside its `SKILL.md`: `eval.yaml`, `instructions/`, `graders/`, `rubrics/`, `solutions/`, and optionally `fixtures/`. These are omitted from the tree above for readability; see [Evaluating Skills](#-evaluating-skills) and [docs/eval-guidelines.md](docs/eval-guidelines.md).

## 🎯 Available Skills

### 1. Adaption AI SDK
**Status:** ✅ Complete

**Description:** Build dataset augmentation pipelines with Adaption's Adaptive Data platform for synthetic data generation and fine-tuning preparation.

**Capabilities:**
- Upload and import datasets (local files, Hugging Face, Kaggle)
- Run augmentation/adaptation jobs with brand controls and recipe specifications
- Hallucination mitigation via web-search grounding
- DPO preference pair generation and deduplication
- Quality evaluation and export via presigned URLs
- Async client support with exponential backoff polling
- Built-in [skillgrade](https://github.com/mgechev/skillgrade) evaluation harness (`eval.yaml`)

**Use Cases:**
- Synthetic data generation for fine-tuning
- Dataset augmentation pipelines
- Grounding-based hallucination reduction on training data

**Reference Files:** 2 guides

---

### 2. API Design
**Status:** ✅ Complete

**Description:** Design and review intuitive, scalable, maintainable HTTP APIs — REST primary, with GraphQL/gRPC covered in passing.

**Capabilities:**
- Resource modeling (nouns, collections, sub-resources, action POSTs)
- HTTP semantics: safe/idempotent methods, PUT vs PATCH vs POST, Idempotency-Key
- Correct status codes and RFC 7807 problem+json error envelopes with stable error codes
- Cursor vs offset pagination, filtering, sorting, sparse fieldsets
- Versioning strategies and backward-compatible evolution (Sunset headers)
- Spec-first OpenAPI 3.x workflow: Spectral linting, ReDoc, contract testing
- Auth (API keys, OAuth2 scopes) and rate limiting (X-RateLimit-* headers)
- A 12-point pre-review checklist for existing APIs

**Use Cases:**
- Designing a new REST API or endpoint set
- Reviewing API specs and PRs that change endpoint behavior
- Establishing API design standards for a team
- Writing OpenAPI definitions and contract tests

**Reference Files:** 3 guides

---

### 3. Database Design
**Status:** ✅ Complete

**Description:** Design relational database schemas (and decide when NoSQL fits) that stay maintainable and fast.

**Capabilities:**
- Requirements analysis: entities, relationships, cardinality, CRUD vs reporting reads
- Normalization 1NF–3NF and deliberate denormalization tradeoffs
- Keys: natural vs surrogate, composite, UUID vs bigint, referential actions
- Indexing: B-tree, composite/covering/partial indexes, reading EXPLAIN, write amplification
- DB-enforced constraints and transactions with isolation levels
- Zero-downtime migrations (Alembic, Prisma, Flyway) with expand-contract
- SQL vs NoSQL decision table (relational, document, wide-column, graph)
- Practical patterns: JSON columns, full-text search, N+1 prevention

**Use Cases:**
- Designing schemas for new features or applications
- Choosing between SQL and NoSQL stores
- Planning safe, zero-downtime schema migrations
- Optimizing slow queries with EXPLAIN and targeted indexes

**Reference Files:** 3 guides

---

### 4. Domain-Driven Design (DDD)
**Status:** ✅ Complete

**Description:** Comprehensive DDD skill for building software that reflects deep understanding of business domains.

**Capabilities:**
- Strategic design (bounded contexts, subdomains, context maps)
- Tactical design (entities, value objects, aggregates, repositories)
- Architecture patterns (Hexagonal, CQRS, Event Sourcing, Clean Architecture)
- Event Storming facilitation
- Language-specific implementations (Python with Pydantic/FastAPI, TypeScript with NestJS)
- DDD code review guidance

**Use Cases:**
- Designing new systems with DDD principles
- Refactoring existing codebases toward DDD
- Generating code scaffolding (entities, aggregates, repositories)
- Performing code reviews with a DDD lens

**Reference Files:** 7 comprehensive guides

---

### 5. genai-tk — GenAI & Agentic Toolkit
**Status:** ✅ Complete

**Description:** YAML-driven wrapper over LangChain, LangGraph, and 100+ LLM providers. Inversion-of-control layer where profiles in YAML drive factories that produce LangChain runtime objects. Synced with upstream [tclatos/genai-tk](https://github.com/tclatos/genai-tk) at commit `7fbd7a3`, including a table of places where the upstream docs and code disagree.

**Capabilities:**
- `model_id@provider` LLM and embeddings factories, reasoning effort, gateway routing, decision models
- `config/app_conf.yaml` with `:merge` / `:profiles` overlays and `GENAITK_PROFILE` switching
- Unified `agents:` profiles for two harnesses: LangChain (`react`, `deep`, `custom`) and DeerFlow
- `LangchainAgent` and the harness-agnostic `create_harness` event stream
- CodeAct agents, middlewares, deepagents skills, OpenSandbox Docker sandbox, MCP client and server
- `RetrieverFactory` with six retriever types (vector, BM25, ensemble, reranked, pg_hybrid, zero_entropy)
- YAML workflows on Prefect, ingest routing, and markdownize document conversion
- BAML structured extraction
- CLI scaffolding with `cli init` and a generated justfile

**Use Cases:**
- Building production-grade agent systems with YAML configuration
- Multi-step planning with deep agents and sandboxed execution
- Deep web research with DeerFlow
- Document ingestion, RAG, and structured extraction pipelines

**Reference Files:** 7 guides

---

### 6. LangChain
**Status:** ✅ Complete

**Description:** Build AI agents with LangChain v1 and LangGraph. Verified against `langchain` 1.4.6, `langgraph` 1.2, `deepagents` 0.7 and `managed-deepagents` 0.9 using the current docs.

**Capabilities:**
- `create_agent`, `init_chat_model`, model profiles and `ToolRuntime` context
- Middleware: summarization, retries, fallbacks, call limits, PII, human-in-the-loop, and custom hooks
- Structured output with `ToolStrategy` / `ProviderStrategy`
- Built-in MCP client (`langchain.mcp.MCPAdapter`) replacing `langchain-mcp-adapters`
- Event streaming (`stream_events(version="v3")`), checkpointers, stores and durability modes
- Multi-agent patterns (subagents, handoffs, routers, skills)
- Deep Agents harness: backends, permissions, sandboxes, code interpreter, sync/fork/async subagents
- Managed Deep Agents on LangSmith (`mda` CLI): memory, identity, connections, sandboxes, schedules
- RAG with provider packages, `langchain_text_splitters` and `langchain_classic` retrievers
- Migration table from 0.x APIs (`create_react_agent`, chains, MCP interceptors)

**Reference Files:** 5 guides

---

### 7. Microservices Patterns
**Status:** ✅ Complete

**Description:** Decompose systems into microservices and apply the canonical distributed-systems patterns — or decide a modular monolith is the better call.

**Capabilities:**
- Microservices vs modular monolith decision criteria
- Decomposition by bounded context/subdomain with database-per-service
- Sync (REST/gRPC) vs async (events/messages) communication choices
- Sagas (choreography & orchestration), compensating actions, and the outbox pattern
- CQRS and event sourcing with honest cost-benefit
- API gateway, BFF, and service discovery
- Resilience: timeouts, retries with jitter, circuit breakers, bulkheads, idempotent consumers
- Observability: structured logs, metrics, OpenTelemetry distributed tracing
- Contract testing with Pact and independent deployability (canary releases)

**Use Cases:**
- Splitting a monolith into services (strangler fig extraction)
- Designing service boundaries and communication flows
- Implementing distributed transactions safely (sagas + outbox)
- Hardening services against partial failure

**Reference Files:** 3 guides

---

### 8. Performance Optimization
**Status:** ✅ Complete

**Description:** Find and fix performance problems by measuring first — a systematic profile → fix → re-measure workflow for backend, frontend, and network.

**Capabilities:**
- Profiling: cProfile, py-spy, Chrome DevTools, Node --cpu-prof, perf
- Metrics: p50/p95/p99 latency, throughput, budgets, SLOs; load testing with k6/locust
- Backend: EXPLAIN-driven query tuning, N+1 fixes, caching (Redis), connection pooling
- Frontend: code splitting, lazy loading, image optimization, memoization, virtualization
- Network: CDNs, compression, HTTP/2/3, prefetch/preconnect
- Memory leaks and concurrency models (threads/async/workers, GIL-aware)

**Use Cases:**
- Diagnosing slow endpoints and page loads with before/after proof
- Establishing performance budgets and load-testing against them
- Fixing N+1 queries, missing indexes, cache stampedes, and pool exhaustion
- Reducing bundle size and render-blocking resources

**Reference Files:** 3 guides

---

### 9. prime-rl — Post-Training at Scale
**Status:** ✅ Complete

**Description:** Post-train LLMs with PrimeIntellect's [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl): asynchronous RL, SFT, distillation and evals on verifiers environments. Built from the upstream `skills/` runbooks and checked against the config classes at commit `dd72f0f`, including a table of places where the upstream docs and code disagree.

**Capabilities:**
- Install with uv, opt-in environment packages, and optional kernel/DeepEP extras
- pydantic-config TOML + CLI composition (`@ file.toml`, dotted overrides, `--dry-run`)
- `uv run rl` / `sft` / `inference` / `eval` entrypoints and run-directory layout
- Algorithms: GRPO, MaxRL, RAE, hierarchical GRPO, OPD, SFT-distill, OPSD, ECHO
- RL losses (IPO default, IcePop, PPO, CISPO, custom), length penalties, curricula, renderers
- Checkpointing, resume, fork, and DCP → HF safetensors / FP8 export
- Metrics, logs, dashboard and an agent check-in runbook for live runs
- Multi-node SLURM, FSDP/EP/CP parallelism, LoRA, weight broadcast, disaggregated inference

**Use Cases:**
- Writing and validating `rl.toml` / `sft.toml` / `eval.toml` configs
- Choosing and tuning a post-training algorithm for a verifiable-reward task
- Launching, monitoring, resuming and exporting training runs
- Benchmarking checkpoints on verifiers environments

**Reference Files:** 7 guides

---

### 10. Scalingo
**Status:** ✅ Complete

**Description:** Deploy and operate web applications on Scalingo, a European (French) Platform-as-a-Service with Heroku-compatible buildpacks and sovereign cloud regions.

**Capabilities:**
- App creation, deployment, and scaling via CLI and git
- Managed database addons (PostgreSQL, MySQL, MongoDB, Redis, OpenSearch, InfluxDB)
- Horizontal and vertical container scaling with autoscaler support
- `scalingo.json` manifest and review app configuration
- Terraform Infrastructure-as-Code provider
- Sovereign region (`osc-secnum-fr1`) on SecNumCloud-qualified Outscale infrastructure for French public-sector workloads
- Migration guidance from Heroku
- Deployment recipes from the official Tutorials Hub (Keycloak, PostgREST, WordPress, n8n, Metabase, ModSecurity, AI demos)

**Use Cases:**
- Deploying web apps to a European sovereign PaaS
- Managing production databases and background workers
- Automating infrastructure with Terraform
- Meeting French public-sector compliance requirements (HDS, SecNumCloud infrastructure)

**Reference Files:** 8 guides

---

### 11. Security Best Practices
**Status:** ✅ Complete

**Description:** Practical, actionable security review and hardening guidance for Python, JavaScript/TypeScript, and Go code — engineering fixes, not a compliance checklist.

**Capabilities:**
- Threat modeling with STRIDE in five minutes
- OWASP Top 10 mapped to code-level fixes (injection, XSS, IDOR, CSRF, SSRF)
- Authentication: argon2id/bcrypt hashing, sessions, MFA, login rate limiting
- Authorization: RBAC/ABAC and object-level access control (IDOR prevention)
- OAuth2/OIDC/JWT: flows, signature verification pitfalls, token storage
- Secrets management and supply-chain scanning (pip-audit, npm audit, gitleaks, SBOM)
- Secure defaults: TLS, CSP, security headers, CORS, cookie flags
- Language cheat-sheets for Python, JS/TS, and Go pitfalls

**Use Cases:**
- Security reviews and prioritized vulnerability reports
- Hardening auth and fixing OWASP-class vulnerabilities
- Secure-by-default development of new endpoints
- Setting up dependency scanning in CI

**Reference Files:** 2 guides

---

### 12. SmolAgents
**Status:** ✅ Complete

**Description:** Build AI agents with Hugging Face's minimalist SmolAgents framework.

**Capabilities:**
- CodeAgent and ToolCallingAgent creation
- Custom tool development
- MCP (Model Context Protocol) integration
- Multi-agent systems
- Secure code execution (E2B, Docker, Blaxel)
- Model configuration (HF Inference, LiteLLM, Transformers, Ollama)
- Agentic RAG and text-to-SQL pipelines
- Web browsing agents

---

### 13. Testing Patterns
**Status:** ✅ Complete

**Description:** A language-agnostic playbook for planning, writing, and maintaining automated tests — with worked examples in Python and JavaScript/TypeScript.

**Capabilities:**
- Test pyramid/trophy: unit vs integration vs e2e placement decisions
- TDD red-green-refactor workflow and when to skip it
- Unit test design: arrange-act-assert, given-when-then naming, fakes vs stubs vs mocks, dependency injection
- Integration tests with testcontainers, database seeding, transaction rollback
- Playwright/Cypress e2e testing for critical journeys
- Property-based testing (Hypothesis, fast-check), fixtures, factories, parametrized tests
- Coverage and mutation testing, flaky-test triage in CI

**Use Cases:**
- Writing or planning tests for a feature
- Reviewing test suites for coverage and reliability
- Setting up integration/e2e infrastructure (testcontainers, Playwright, CI sharding)
- Debugging intermittent CI failures

**Reference Files:** 3 guides

---

### 14. UI Skills
**Status:** ✅ Complete

**Description:** Opinionated constraints for building better interfaces with agents. Ensures accessibility, performance, and consistent design quality in AI-generated UI code.

**Capabilities:**
- Tailwind CSS and motion/react animation guidelines
- Accessible component primitives (Base UI, React Aria, Radix)
- Interaction best practices (AlertDialog for destructive actions, structural skeletons for loading)
- Animation constraints (compositor-only props, 200ms limit, reduced-motion support)
- Typography and layout rules (text-balance, tabular-nums, z-index scale)
- Performance guidelines (no large blur, no will-change outside animations)

**Use Cases:**
- Reviewing agent-generated UI for quality and accessibility
- Ensuring consistent Tailwind CSS usage
- Preventing common AI-generated UI anti-patterns

---

### 15. Unsloth Training on HF Jobs
**Status:** ✅ Complete

**Description:** Fine-tune LLMs and VLMs using Unsloth on Hugging Face on-demand cloud GPUs with UV scripts.

**Capabilities:**
- VLM fine-tuning (Qwen3-VL, Gemma 3) with image + message datasets
- Continued pretraining and domain adaptation
- LoRA fine-tuning with configurable rank and learning rate
- Trackio monitoring integration
- Automated dependency management via UV scripts

**Use Cases:**
- Fine-tuning vision-language models on custom datasets
- Domain adaptation with continued pretraining
- Running GPU training jobs without local hardware

---

## 🚀 Usage

### For AI Coding Agents

Each skill can be loaded by agents to enhance their capabilities:

1. **Load a skill:** Reference the `SKILL.md` file in the appropriate skill directory
2. **Access references:** Each skill includes detailed reference documentation in its `references/` folder
3. **Apply patterns:** Follow the workflows and examples provided in the skill documentation

### For Developers

**Clone the repository:**
```bash
git clone https://github.com/svngoku/coding-agents-skills.git
cd coding-agents-skills
```

**Browse skills:**
```bash
# View available skills
ls skills/

# Read a specific skill
cat skills/ddd/SKILL.md

# Explore reference materials
ls skills/ddd/references/
```

### Integration Examples

#### With SmolAgents
```python
from pathlib import Path

from smolagents import CodeAgent, InferenceClientModel

ddd_skill = Path("skills/ddd/SKILL.md").read_text()

agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    additional_authorized_imports=["pydantic", "typing"],
    instructions=ddd_skill,  # appended to the agent's system prompt
)

result = agent.run(
    "Design a bounded context for an e-commerce order management system "
    "following DDD principles."
)
```

#### With LangChain
```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

@tool
def load_skill(skill_path: str) -> str:
    """Load a skill's SKILL.md content by path."""
    return open(skill_path).read()

agent = create_agent(
    model=init_chat_model("anthropic:claude-sonnet-5", temperature=0),
    system_prompt="You are a software architecture advisor.",
    tools=[load_skill],
)

result = agent.invoke({
    "messages": [{"role": "user",
                  "content": "Review this code for DDD compliance "
                             "using the DDD skill."}]
})
```

## 📋 Skill Template

Each skill follows a consistent structure (the authoritative version lives in [AGENTS.md](AGENTS.md)):

### SKILL.md Format
```markdown
---
name: skill-name            # kebab-case, must match the directory name
description: >
  <trigger sentence>. Use this skill whenever the user mentions <keywords>,
  wants to <actions>, or works with <tools/frameworks>. Also trigger for
  <related terms>.
---

# Human-Readable Skill Name

[Brief 1-paragraph overview — what problem this skill solves]

## Quick Reference
[Table linking to reference files — only if references/ exists]

## Core Workflow / Usage
[Step-by-step instructions for the most common tasks]

## <Topic Sections>
[Concrete examples, code snippets, configuration samples]

## Anti-Patterns to Avoid
[Common mistakes]

## When to Use / Not Use
[Decision criteria]
```

### References Structure
- Each skill has a `references/` directory (when applicable)
- Reference files are in Markdown format
- Cover specific aspects of the skill in depth
- Include code examples in relevant languages

## 🌐 Gallery Site

The collection has a browsable gallery: <https://svngoku.github.io/coding-agents-skills/>

It is a Vite + React + TypeScript app styled with Tailwind CSS v4 and
[shadcn/ui](https://ui.shadcn.com/) on Base UI primitives, published with GitHub Pages. Python
remains the content pipeline; the app only renders.

```bash
# content: reads skills/ and README.md, writes the payload the app fetches
python3 scripts/build-site.py          # writes site/public/gallery.json
python3 scripts/build-site.py --check  # fails if that payload is out of date

# app: Vite + React
cd site
npm ci
npm run dev      # local dev server
npm run build    # static bundle in site/dist
npm run preview  # serve the built bundle
```

`scripts/build-site.py` reads each `SKILL.md` frontmatter, the per-skill sections of this README,
the `references/` and `scripts/` folders and every `eval.yaml`, renders the Markdown to HTML, and
writes a single JSON payload. The app never parses Markdown, so the generator stays
standard-library only and the `--check` gate keeps `site/public/gallery.json` honest.

The UI is assembled from shadcn/ui components (`Card`, `Button`, `Badge`, `Dialog`, `Select`,
`ToggleGroup`, `Tooltip`, `ScrollArea`, `Table`, `InputGroup`, `Alert`, `Empty`, `Separator`,
`Skeleton`) and renders:

- **Cards** — title, category, description, capability tags, guide/line/eval counts and actions.
  Every card has the same anatomy and its footer is pinned to the card bottom, so the "Details"
  and "Read" actions line up across a row instead of leaving dead space underneath. Phones keep a
  single badge row; the extra counts appear from the `sm` breakpoint.
- **Details dialog** — a segmented `ToggleGroup` switches between guides, scripts and use cases,
  with full (never truncated) reference titles, status/line/eval counts in the header, one click
  to open a document and links to the GitHub source.
- **Document viewer** — `SKILL.md`, every reference guide and example script rendered in a
  `ScrollArea` behind a grouped shadcn `Select` that jumps between files. highlight.js is bundled
  locally (14 languages) and colours fenced code with the gallery palette: amber keywords,
  emerald strings, rose literals, no blue.
- **Matrix** — artifact coverage per skill (guides, scripts, tasks, graders, rubrics, solutions),
  available through the `ToggleGroup` card/matrix view switch.
- **Filters** — `InputGroup` search with an inline clear button, category chips with `aria-pressed`,
  shadcn `Select` sorting by name, guide count, documentation size or category, dark/light theme
  and `#skill-<name>` deep links.

Both dialogs are lazy-loaded, so the Markdown renderer and highlight.js stay out of the initial
bundle (the app entry stays under 455 kB, the dialogs load on first open). Icon-only controls use
`Tooltip` and `aria-label`s, and the light theme uses emerald-700 so small primary-coloured text
clears WCAG AA contrast against white.

The app follows this repo's own [`ui` skill](skills/ui/SKILL.md) and the
[ui-skills.com](https://www.ui-skills.com/) polish rules: one accent colour (emerald), a neutral
(never blue) dark palette, `text-balance`/`text-pretty`/`tabular-nums` typography, visible focus
rings and `aria-label`s on icon-only buttons. It is responsive from 320px up: one card column on
phones, two on tablets, three on wide screens, and the dialogs switch to a full-height layout on
small viewports.

### Typography

Body copy is set in **Aspekta**, a sans-serif by Ivo Dolenc released under the SIL Open Font
License 1.1. The variable `woff2` (weights 100–900) is bundled at
`site/src/fonts/AspektaVF.woff2` next to its licence (`Aspekta-LICENSE.txt`).

Headings and card titles are set in **NB International Pro**. The typeface is commercial, so the
licensed files are not committed: put them under `site/src/fonts/` (git-ignored) and add the
`@font-face` rule in `site/src/index.css`, or point the build at a stylesheet that serves them.

```bash
VITE_NB_FONT_CSS=https://cdn.example.com/nb-fonts.css npm run build
```

Until the licensed files are available, titles fall back to Aspekta, so the page keeps a coherent
look; if both are missing it falls back to the system sans stack.

`.github/workflows/pages.yml` verifies the committed payload, rebuilds it, builds the Vite app and
deploys `site/dist` to GitHub Pages. The build fails when `site/public/gallery.json` no longer
matches `skills/`.

One-time setup on a fresh fork or clone: enable Pages under **Settings → Pages → Build and
deployment → Source: GitHub Actions**. The workflow's `configure-pages` step can create the site
itself, but only for a token that is allowed to administer the repository; the default
`GITHUB_TOKEN` cannot, so the first run fails with "Resource not accessible by integration"
until Pages exists. Once the site is enabled, pushes deploy normally.

## 🧪 Evaluating Skills

Skills are evaluated with [skillgrade](https://github.com/mgechev/skillgrade#readme) — "unit tests for your agent skills". A real coding agent runs a task against the skill, and its output is scored by a **deterministic grader** (static API-surface checks) plus an **LLM rubric** (approach quality). The repo standardizes on the **code-generation archetype** (deterministic 0.7 / rubric 0.3), so evaluations are fast and hermetic — no live services required.

### Coverage

| Skill | Harness | Status |
|-------|---------|--------|
| adaption-ai | ✅ `eval.yaml` | ready |
| langchain | ✅ `eval.yaml` | ready |
| smolagents | ✅ `eval.yaml` | ready |
| genai-tk | ✅ `eval.yaml` | ready |
| unsloth-hf-jobs | ✅ `eval.yaml` | ready |
| database-design | ✅ `eval.yaml` | ready |
| api-design | ✅ `eval.yaml` | ready |
| security-best-practices | ✅ `eval.yaml` | ready |
| testing-patterns | ✅ `eval.yaml` | ready |
| performance-optimization | ✅ `eval.yaml` | ready |
| ddd | ✅ `eval.yaml` | ready |
| microservices-patterns | ✅ `eval.yaml` | ready |
| ui | ✅ `eval.yaml` | ready |
| scalingo | ✅ `eval.yaml` | ready |
| prime-rl | ✅ `eval.yaml` | ready |

### Running locally (no CI)

```bash
npm i -g skillgrade

./scripts/run-evals.sh                  # smoke-test every skill with an eval.yaml
./scripts/run-evals.sh langchain        # one skill
./scripts/run-evals.sh --mode=reliable  # 15 trials (regression: 30)
./scripts/run-evals.sh --validate       # verify graders against reference solutions
```

Reports land in `.evals/<skill>/` (gitignored); `--ci` fails the run when a skill drops below its threshold. See [docs/eval-guidelines.md](docs/eval-guidelines.md) for the layout convention and authoring rules.

## 🤝 Contributing

Contributions are welcome! To add a new skill:

1. **Fork the repository**
2. **Create a new skill directory** under `skills/`
3. **Follow the skill template:**
   - Create `SKILL.md` with the standard structure (including YAML frontmatter with `name` and `description`)
   - Add `references/` directory with detailed documentation when needed
   - Include practical examples and code samples
4. **Submit a pull request**

### Skill Guidelines
- Skills should be atomic and focused on a single domain
- Include both conceptual explanations and practical examples
- Provide language-specific implementations where relevant
- Document anti-patterns and common mistakes
- Keep reference files modular and cross-referenced
- Use YAML frontmatter with `name` and `description` for machine readability
- See `AGENTS.md` for the full convention checklist

## 📚 Roadmap

- [x] Domain-Driven Design skill
- [x] SmolAgents skill
- [x] LangChain skill
- [x] Adaption AI SDK skill
- [x] genai-tk skill
- [x] Scalingo deployment skill
- [x] UI best practices skill
- [x] Unsloth fine-tuning skill
- [x] Testing patterns skill
- [x] API design skill
- [x] Database design skill
- [x] Microservices patterns skill
- [x] Security best practices skill
- [x] Performance optimization skill
- [x] prime-rl post-training skill
- [ ] MCP server authoring skill
- [ ] RAG patterns skill (retrieval architecture deep dive)
- [ ] LLM evaluation & observability skill (Langfuse, evals)
- [ ] Next.js / React framework patterns skill
- [ ] Kubernetes operations skill

## 🔗 Related Projects

- [Hugging Face SmolAgents](https://github.com/huggingface/smolagents) - Minimalist AI agent framework
- [LangChain](https://github.com/langchain-ai/langchain) - Building applications with LLMs
- [genai-tk](https://github.com/tclatos/genai-tk) - YAML-driven GenAI toolkit
- [Model Context Protocol](https://modelcontextprotocol.io/) - Standard for connecting AI systems
- [Scalingo](https://scalingo.com/) - European Platform-as-a-Service
- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) - Asynchronous RL post-training at scale
- [Adaption](https://docs.adaptionlabs.ai/) - Synthetic data augmentation platform
- [OWASP Top 10](https://owasp.org/Top10/) - Web application security risks
- [Pact](https://pact.io/) - Consumer-driven contract testing
- [OpenTelemetry](https://opentelemetry.io/) - Distributed tracing and observability
- [Playwright](https://playwright.dev/) - End-to-end browser testing

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 👤 Author

**svngoku**
- GitHub: [@svngoku](https://github.com/svngoku)

## 🙏 Acknowledgments

- Inspired by the need for reusable, production-ready AI agent skills
- Built with insights from software architecture patterns and modern AI frameworks
- Community feedback and contributions welcome

---

**Note:** This is an evolving collection. Skills are added and updated based on practical needs in AI-assisted software development.

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
- Deploy and operate applications on cloud platforms (Scalingo)
- Follow best practices in code generation, refactoring, and UI development
- Integrate seamlessly with development workflows

Each skill is self-contained with comprehensive documentation, reference materials, and practical examples.

## 🗂️ Project Structure

```
coding-agents-skills/
├── skills/
│   ├── adaption-ai/                    # Adaption AI SDK for synthetic data augmentation
│   │   ├── SKILL.md
│   │   ├── eval.yaml                   # skillgrade evaluation harness
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
│   ├── genai-tk-skill/                 # GenAI Toolkit (YAML-driven agent framework)
│   │   ├── SKILL.md
│   │   ├── genai-tk-skill.skill        # Packaged skill archive (zip)
│   │   └── references/
│   │       ├── agents.md
│   │       ├── baml-structured.md
│   │       ├── cli-and-init.md
│   │       ├── configuration.md
│   │       └── rag.md
│   │
│   ├── langchain/                      # LangChain framework
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── langgraph.md
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
│   ├── scalingo/                       # Scalingo European PaaS deployment
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── addons-databases.md
│   │       ├── buildpacks.md
│   │       ├── cli-reference.md
│   │       ├── deployment.md
│   │       ├── manifest-review-apps.md
│   │       ├── scaling-autoscaler.md
│   │       └── terraform-iac.md
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
│   └── run-evals.sh                 # local eval runner (no CI)
└── .gitignore                       # ignores .evals/ output
```

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

**Description:** YAML-driven wrapper over LangChain, LangGraph, and 100+ LLM providers. Inversion-of-control layer where profiles in YAML drive factories that produce LangChain runtime objects.

**Capabilities:**
- `model_id@provider` LLM and embeddings factories
- Four bundled agent frameworks (ReAct, Deep, Deer-flow, SmolAgents)
- Unified `LangchainAgent` entry point with profile-based configuration
- `RetrieverFactory` with six retriever types (vector, BM25, ensemble, reranked, pg_hybrid, zero_entropy)
- BAML structured extraction
- OpenSandbox Docker integration for secure code execution
- MCP server registry and SkillsMiddleware for on-demand domain knowledge
- CLI scaffolding with `cli init`

**Use Cases:**
- Building production-grade agent systems with YAML configuration
- Multi-step planning with Deep agents and sandboxed execution
- Deep web research with Deer-flow
- Code-first automation with SmolAgents

**Reference Files:** 5 guides

---

### 6. LangChain
**Status:** ✅ Complete

**Description:** Build AI applications with the LangChain framework.

**Capabilities:**
- Chain construction and composition
- Memory management and stateful agents (LangGraph checkpointing)
- Agent creation and orchestration
- Tool integration with `ToolRuntime` context
- Structured output and MCP integration
- RAG (Retrieval-Augmented Generation) patterns

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

### 9. Scalingo
**Status:** ✅ Complete

**Description:** Deploy and operate web applications on Scalingo, a European (French) Platform-as-a-Service with Heroku-compatible buildpacks and sovereign cloud regions.

**Capabilities:**
- App creation, deployment, and scaling via CLI and git
- Managed database addons (PostgreSQL, MySQL, MongoDB, Redis, OpenSearch, InfluxDB)
- Horizontal and vertical container scaling with autoscaler support
- `scalingo.json` manifest and review app configuration
- Terraform Infrastructure-as-Code provider
- SecNumCloud-qualified region (`osc-secnum-fr1`) for French public-sector workloads
- Migration guidance from Heroku

**Use Cases:**
- Deploying web apps to a European sovereign PaaS
- Managing production databases and background workers
- Automating infrastructure with Terraform
- Meeting French public-sector compliance requirements (HDS, SecNumCloud)

**Reference Files:** 7 guides

---

### 10. Security Best Practices
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

### 11. SmolAgents
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

### 12. Testing Patterns
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

### 13. UI Skills
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

### 14. Unsloth Training on HF Jobs
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
from smolagents import CodeAgent, HfApiModel

# Load DDD skill for architectural guidance
agent = CodeAgent(
    tools=[],
    model=HfApiModel(),
    additional_authorized_imports=["pydantic", "typing"]
)

result = agent.run(
    "Design a bounded context for an e-commerce order management system "
    "following DDD principles. Use the DDD skill reference."
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
    model=init_chat_model("claude-sonnet-4-5-20250929", temperature=0),
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

Each skill follows a consistent structure:

### SKILL.md Format
```markdown
---
name: skill-name
description: Detailed description of when and how to use this skill
---

# Skill Name

## Overview
[Brief introduction]

## Quick Reference
[Table linking to reference files]

## Core Workflow
[Step-by-step usage guide]

## Implementation Guidelines
[Concrete examples and patterns]

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
| api-design | ⏳ planned | — |
| security-best-practices | ⏳ planned | — |
| testing-patterns | ⏳ planned | — |
| performance-optimization | ⏳ planned | — |
| ddd | ⏳ planned | — |
| microservices-patterns | ⏳ planned | — |
| ui | ⏳ planned | — |
| scalingo | ⏳ planned | — |

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

# Mastra Version and Source Ledger

This skill is an independent, documentation-based guide for Mastra. It is not an official Mastra
skill and contains no copied logo or implied endorsement. Research was performed on **2026-11-08**.
Current documentation is canonical; inspect the exact installed package before using version-sensitive
APIs.

## Version Baseline

| Item | Verified state |
| --- | --- |
| Stable core | `@mastra/core` **1.82.0** from the npm `latest` metadata |
| Repository development version | `1.83.0-alpha.1`; not presented as stable |
| Node requirement | `>=22.13.0` for core 1.82.0 |
| Core license | Apache-2.0 |
| Zod peer range | `^3.25.0 || ^4.0.0` |
| Knowledge | Introduced as **alpha** in core 1.82.0 |
| Language focus | TypeScript |
| Model IDs | Examples are real at research time; verify current catalog before use |

Version metadata: <https://registry.npmjs.org/@mastra/core/latest>.
Development package evidence: <https://github.com/mastra-ai/mastra/blob/main/package.json>.
Model catalog: <https://mastra.ai/models>.

## How to Update This Skill

1. Read <https://mastra.ai/llms.txt> and the exact current reference pages, not training-memory snippets.
2. Check npm stable metadata and the core package's Node/peer requirements.
3. Search this skill for old imports, callback signatures, model IDs, and alpha/stable statements.
4. Update examples and the source ledger together; record any docs/API disagreement.
5. Run the offline harness, structural checks, and gallery build. Do not describe those checks as live
   Mastra compilation or model evaluation.

No paid provider or model calls were made while creating this skill. Documentation verification and
static grading do not establish that every example compiles against every supported provider.

## Agents and Application Registration

- Overview: <https://mastra.ai/docs/agents/overview.md>
- Tools: <https://mastra.ai/docs/agents/tools.md>
- Human in the loop: <https://mastra.ai/docs/agents/human-in-the-loop.md>
- Guardrails: <https://mastra.ai/docs/agents/guardrails.md>

Verified distinctions: `Agent` comes from `@mastra/core/agent`; the model router uses `provider/model`;
`structuredOutput.schema` yields `response.object`; registered agents can be retrieved by ID;
`createTool` uses `execute(inputData, context)`; sensitive tools support `requireApproval`.

## Memory and Workflows

- Memory overview: <https://mastra.ai/docs/memory/overview.md>
- Workflow overview: <https://mastra.ai/docs/workflows/overview.md>
- Control flow: <https://mastra.ai/docs/workflows/control-flow.md>
- Suspend and resume: <https://mastra.ai/docs/workflows/suspend-and-resume.md>

Verified distinctions: message memory requires storage; stable resource/thread scope is mandatory;
Observational Memory changes raw-history behavior and can require `memory.settled()`; workflow steps
use `execute({ inputData, ... })`; runs have success, failed, suspended, tripwire, or paused status;
persistent snapshots are required for restart recovery.

## Retrieval

- Knowledge overview: <https://mastra.ai/docs/knowledge/overview.md>
- Chunking and embedding: <https://mastra.ai/reference/rag/chunking-and-embedding.md>
- Retrieval: <https://mastra.ai/reference/rag/retrieval.md>
- `rerank()`: <https://mastra.ai/reference/rag/rerank.md>
- libSQL vectors: <https://mastra.ai/reference/vectors/libsql.md>
- Metadata filters: <https://mastra.ai/reference/rag/metadata-filters.md>
- Vector query tool: <https://mastra.ai/reference/tools/vector-query-tool.md>

Verified distinctions: `MDocument` and `createVectorQueryTool` come from `@mastra/rag`; embedding uses
`ModelRouterEmbeddingModel` with AI SDK `embed`/`embedMany`; current `rerank()` has positional
`results, query, model, options` arguments; Knowledge's alpha status and isolation limitations matter.

## Composition, Evaluation, and Operations

- Subagents: <https://mastra.ai/docs/subagents.md>
- MCP: <https://mastra.ai/docs/connections/mcp.md>
- Evaluation overview: <https://mastra.ai/docs/evals/overview.md>
- Vitest integration: <https://mastra.ai/docs/evals/vitest-integration.md>
- Tracing overview: <https://mastra.ai/docs/observability/tracing/overview.md>
- Mastra server: <https://mastra.ai/docs/deployment/mastra-server.md>

Verified distinctions: parent `agents: { ... }` is the current supervisor pattern; MCP uses
`MCPClient`/`MCPServer` from `@mastra/mcp`; request-specific tools use `listToolsets()` and per-request
cleanup; deterministic eval checks, built-in scorers, and Vitest integration are separate layers;
`Observability` comes from `@mastra/observability`; build output is `.mastra/output` and health is
`GET /health`.

## Security Interpretation

The skill applies general application-security principles to documented Mastra extension points:
server-derived identity and tenant scope, object-level authorization, least privilege, exact-argument
approval, idempotency, bounded execution, untrusted external content, secret minimization, and tested
recovery. Guardrail processors and MCP annotations supplement these controls; they do not replace them.

For implementation work, consult:

- Installed package changelog and migration guides
- Exact API reference for the selected storage, vector, model, and deployer integrations
- Provider documentation for model availability, idempotency, limits, retention, and privacy
- Application threat model and regulatory requirements

When documentation and installed behavior disagree, report the discrepancy, use the installed package's
types/source as local evidence, and avoid presenting an unverified workaround as stable API guidance.

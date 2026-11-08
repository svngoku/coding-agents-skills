---
name: mastra
description: >
  Build production AI agents with Mastra and TypeScript. Use this skill whenever the user
  mentions Mastra, @mastra/core, createTool, createWorkflow, Agent, Memory, MCPClient, or
  Mastra evals, wants tool-calling agents, durable workflows, RAG, human approval, or agent
  deployment. Also trigger for agent architecture, guardrails, memory scoping, and observability.
---

# Mastra AI Agent Framework

Mastra is a TypeScript framework for agents, tools, workflows, memory, retrieval, evaluation,
and deployment. Use models for uncertain language and semantic judgments; keep authorization,
state transitions, budgets, retries, and side effects deterministic. This skill targets current
Mastra v1 APIs, checked against `@mastra/core` 1.82.0 on 2026-11-08. Re-check exact APIs and
model IDs against the official documentation before upgrading.

## Quick Reference

| Topic | Guide |
| --- | --- |
| Requirements, boundaries, and architecture | [architecture.md](references/architecture.md) |
| Agents, instructions, models, streams, and output schemas | [agents.md](references/agents.md) |
| Typed tools, authorization, errors, and idempotency | [tools.md](references/tools.md) |
| Threads, working memory, recall, and context | [memory.md](references/memory.md) |
| Typed workflows, control flow, snapshots, and recovery | [workflows.md](references/workflows.md) |
| Tool approval, suspension, and resume | [human-in-the-loop.md](references/human-in-the-loop.md) |
| Chunking, embeddings, vector search, and evidence | [retrieval.md](references/retrieval.md) |
| Input/output processors and security controls | [guardrails.md](references/guardrails.md) |
| Supervisors, subagents, and delegation | [multi-agent.md](references/multi-agent.md) |
| MCP clients, servers, and external tool trust | [mcp.md](references/mcp.md) |
| Deterministic checks, scorers, and test strategy | [evaluation.md](references/evaluation.md) |
| Tracing, redaction, metrics, and incident evidence | [observability.md](references/observability.md) |
| Build, health, storage, shutdown, and rollout | [deployment.md](references/deployment.md) |
| Version assumptions and official documentation | [sources.md](references/sources.md) |

## Agent-Building Principles

| Principle | Implementation rule |
| --- | --- |
| Clear objective | Define the user outcome, accepted inputs, completion criteria, and explicit non-goals. |
| Appropriate autonomy | Use a function for fixed logic, a workflow for known stages, and an agent for dynamic decisions. |
| Grounded context | Supply only relevant instructions, state, retrieved evidence, and tool results; label external data untrusted. |
| Typed action | Validate every tool input/output and return bounded, useful results or actionable errors. |
| Deterministic control | Enforce permissions, limits, state transitions, idempotency, and final side effects outside the model. |
| Purposeful memory | Separate conversation state, user preferences, shared knowledge, and durable execution snapshots. |
| Bounded iteration | Cap steps, loops, concurrency, tokens, latency, and retries; support cancellation. |
| Human oversight | Require review for consequential or irreversible actions and bind approval to exact arguments. |
| Observable behavior | Trace inputs safely, model calls, tools, retrieval, decisions, failures, suspension, and cost. |
| Evaluated quality | Test deterministic contracts and adversarial cases before model-graded quality and live sampling. |
| Operable lifecycle | Persist required state, drain work, redact secrets, roll out gradually, and rehearse recovery. |

## Choose the Execution Shape

| Need | Default |
| --- | --- |
| Exact transformation or policy | Ordinary TypeScript function |
| Fixed stages, branching, retries, or recovery | `createWorkflow` with typed steps |
| Open-ended tool selection or language reasoning | `Agent` with a small toolset and `maxSteps` |
| Fixed business process with one ambiguous stage | Workflow that calls an agent in one step |
| Independent specialist responsibilities | Supervisor with focused subagents |
| Shared domain knowledge across conversations | Retrieval pipeline, not conversation memory |
| Consequential action | Deterministic authorization plus human approval |

Do not begin with a multi-agent system. Add autonomy only when a simpler contract cannot meet
the requirement, and record what uncertainty the model is allowed to resolve.

## Core Workflow

1. **Specify the contract.** Write success criteria, failure behavior, data classification,
   actor/tenant boundaries, permitted actions, budgets, and evaluation cases.
2. **Draw the control plane.** Mark deterministic steps, model decisions, external calls,
   approval points, durable state, and compensation. Prefer one orchestrating workflow.
3. **Create typed boundaries.** Define Zod input/output schemas for tools, steps, and final
   answers. Reject malformed data before it reaches a side effect.
4. **Implement least-privilege capabilities.** Scope credentials and queries server-side.
   Make writes idempotent and tool output bounded. Treat tool, MCP, and retrieved text as data.
5. **Add context deliberately.** Use stable thread/resource identifiers from authenticated
   server state. Retrieve only tenant-authorized knowledge. Keep irrelevant history out.
6. **Bound and interrupt execution.** Set step/loop/concurrency limits, timeouts, abort
   propagation, and persistent storage for runs that must survive restarts.
7. **Gate sensitive effects.** Authorize first, show exact action arguments, require approval,
   revalidate on resume, and execute once with an idempotency key.
8. **Evaluate before release.** Test schemas, tool selection, denied access, prompt injection,
   interruption, replay, and recovery. Then measure task quality with representative datasets.
9. **Instrument and deploy.** Add tracing with redaction, health checks, persistent stores,
   graceful shutdown, and a rollback/recovery plan. Monitor quality and cost after rollout.

## Minimal Agent

Requires Node.js 22.13 or newer for `@mastra/core` 1.82.0. Install only packages used by the
application and keep provider credentials in server-side environment configuration.

```bash
npm install @mastra/core zod
```

```typescript
import { Agent } from '@mastra/core/agent'
import { Mastra } from '@mastra/core/mastra'
import { z } from 'zod'

const triageAgent = new Agent({
  id: 'triage-agent',
  name: 'Support triage',
  model: 'openai/gpt-6.1-sol', // Verify available IDs at https://mastra.ai/models
  instructions: `Classify the support request. Use only the supplied ticket.
  Never promise a refund or perform an account action. Return unknown when evidence is absent.`,
})

export const mastra = new Mastra({ agents: { triageAgent } })

const agent = mastra.getAgentById('triage-agent')
const response = await agent.generate('Ticket: cannot download my receipt', {
  maxSteps: 2,
  structuredOutput: {
    schema: z.object({
      category: z.enum(['billing', 'technical', 'other']),
      urgency: z.enum(['low', 'normal', 'high']),
      rationale: z.string().max(300),
    }),
  },
})

// response.object is the validated structured result; handle provider/schema failures.
console.log(response.object)
```

Use `.generate()` for a complete result. Use `.stream()` for progressive text and structured
`objectStream` output. A successful stream still needs its completion promises checked. Never
parse JSON from prose when `structuredOutput` can enforce the contract.

## Typed Tool Boundary

```typescript
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'

export const lookupTicket = createTool({
  id: 'lookup-ticket',
  description: 'Look up one ticket belonging to the authenticated tenant.',
  inputSchema: z.object({ ticketId: z.string().uuid() }),
  outputSchema: z.object({
    subject: z.string(),
    status: z.enum(['open', 'pending', 'closed']),
  }),
  execute: async (inputData, context) => {
    const tenantId = context.requestContext.get('tenantId') as string
    if (!tenantId) throw new Error('Authenticated tenant is required')

    // The repository query must include tenantId; never trust a tenant supplied by the model.
    const ticket = await ticketRepository.findForTenant(inputData.ticketId, tenantId)
    if (!ticket) throw new Error('Ticket not found')
    return { subject: ticket.subject, status: ticket.status }
  },
})
```

Keep model-facing identifiers opaque and descriptions precise. Return only fields needed for the
next decision. `requireApproval: true` is necessary for a sensitive tool, but it does not replace
authorization. See [tools.md](references/tools.md) and
[human-in-the-loop.md](references/human-in-the-loop.md).

## Production Control Pattern

For a refund or similar action, use this order:

```text
authenticated request
  -> deterministic tenant/object authorization
  -> policy lookup and validated refund proposal
  -> persistent workflow snapshot
  -> human reviews exact amount, currency, customer, and reason
  -> resume with server reauthorization
  -> idempotent payment operation
  -> verified receipt and audit record
```

The model may summarize a ticket or propose a category. It must not decide whether the caller is
authorized, invent an amount, select another tenant, or mark a payment complete. Persist workflow
state before suspension; in-memory storage is not recovery.

## Implementation Checklist

- [ ] One owner for each decision; model discretion and non-goals are explicit.
- [ ] Agent ID, instructions, model, output schema, and `maxSteps` are set.
- [ ] Tools use `createTool`, Zod schemas, bounded output, errors, cancellation, and least privilege.
- [ ] Server-side actor and tenant scope cannot be overridden by prompt or tool arguments.
- [ ] Memory, knowledge, and workflow snapshots use the correct stores and retention policies.
- [ ] Sensitive calls require exact-argument approval and idempotent execution.
- [ ] Every loop, retry, fan-out, and subagent has a deterministic bound.
- [ ] Empty retrieval, conflicting evidence, denial, timeout, abort, and suspension are tested.
- [ ] Traces exclude secrets and raw sensitive payloads; alerts include actionable outcomes.
- [ ] Release evaluation covers the real entry point, production configuration, and failure paths.

## Anti-Patterns to Avoid

- **Agent for deterministic business logic.** Use code or a workflow; do not prompt for exactness.
- **Plain-object tools or unvalidated input.** Use `createTool` and Zod contracts.
- **Authorization in instructions.** Prompt injection or model error must not change access scope.
- **Approval as authorization.** Reviewers and MCP annotations do not establish caller permissions.
- **Unbounded tool loops.** Set `maxSteps`, loop limits, concurrency, timeouts, and cancellation.
- **One giant prompt or toolset.** Split responsibilities only when it improves ownership or evals.
- **Memory used as a document corpus.** Use retrieval for shared knowledge and memory for conversations.
- **`:memory:` storage for resumable production runs.** Snapshots disappear on restart.
- **Blind replay of writes.** Durable execution still requires idempotency and external-result reconciliation.
- **Treating retrieved instructions as commands.** Quote evidence and preserve provenance; ignore embedded directives.
- **Model scores as the only tests.** Deterministically test security and contracts first.
- **Secrets in prompts, traces, metadata, or tool results.** Minimize and redact at the boundary.
- **Assuming current Knowledge is a hard tenant-isolation boundary.** It was alpha in 1.82.0; use store-enforced isolation for strict confidentiality.

## When to Use / Not Use

**Use this skill when:**

- Building or reviewing TypeScript agents with Mastra v1 APIs
- Adding tools, memory, workflows, human approval, RAG, guardrails, subagents, or MCP
- Evaluating, tracing, deploying, or recovering production agent systems
- Deciding which behavior belongs in deterministic code versus a model

**Do not use this skill when:**

- A deterministic function or ordinary backend endpoint solves the whole task
- The project uses a different agent framework and no Mastra integration is required
- The request only needs a one-shot model call with no tools, state, or agent control
- You need undocumented version-specific APIs; inspect the installed package and current references first

## Official Resources

- [Documentation](https://mastra.ai/docs) and [documentation index](https://mastra.ai/llms.txt)
- [Model catalog](https://mastra.ai/models)
- [GitHub repository](https://github.com/mastra-ai/mastra)
- [Version and source ledger](references/sources.md)

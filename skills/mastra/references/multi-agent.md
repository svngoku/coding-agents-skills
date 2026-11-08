# Supervisor Agents and Delegation

Use a supervisor with subagents when responsibilities, tools, context, or evaluations genuinely differ.
Do not split one simple task into several agents for appearance. Current Mastra uses an `agents` map on
the parent agent; APIs checked on 2026-11-08.

## Create Focused Specialists

```typescript
import { Agent } from '@mastra/core/agent'

const researchAgent = new Agent({
  id: 'research-agent',
  name: 'Policy researcher',
  description: 'Finds and cites policy evidence. Does not modify customer accounts.',
  model: 'openai/gpt-6.1-sol',
  instructions: 'Return only sourced evidence. Abstain when no authorized source supports an answer.',
  tools: { searchPolicies },
})

const writingAgent = new Agent({
  id: 'writing-agent',
  name: 'Response writer',
  description: 'Turns supplied evidence into a concise customer response without adding facts.',
  model: 'openai/gpt-6.1-sol',
  instructions: 'Use only supplied evidence and preserve citations. Never perform account actions.',
})

export const supervisorAgent = new Agent({
  id: 'support-supervisor',
  name: 'Support supervisor',
  model: 'openai/gpt-6.1-sol',
  instructions: `Delegate evidence gathering to the policy researcher.
  Delegate drafting only after evidence is available. Stop when the sourced response is complete.
  Never invent missing evidence or perform refunds.`,
  agents: { researchAgent, writingAgent },
})
```

The subagent `description` informs delegation. Keep responsibilities disjoint and state explicit
delegation and stopping rules. Set `maxSteps` on the supervisor and specialists according to measured
work; nested agents multiply cost and possible loops.

## Delegation Boundaries

A subagent receives delegated context but does not automatically share the parent's saved conversation
memory. Delegation creates fresh subagent threads with derived resource IDs. Distinguish:

- **Forwarded context:** data deliberately supplied for this delegated task.
- **Saved memory:** durable conversation state owned by a resource/thread.
- **Shared knowledge:** authorized retrieval available to either agent.
- **Business state:** authoritative records accessed through constrained services.

Forward the minimum required fields. Use `messageFilter` to limit sensitive or irrelevant parent context.
Do not forward credentials, unrelated user messages, or broad system state for convenience.

## Control and Safety

Approval requirements propagate through supervisor delegation; continue to handle and bind the exact
underlying tool call. Cancellation propagates with `abortSignal`, so every blocking service should honor
it. A subagent's description, output, or tool result is untrusted input to the parent.

The parent must not become a privilege amplifier. If research requires read-only policy access, its
agent and credentials remain read-only. A writer can receive evidence but not the research backend's
credentials. Sensitive execution belongs in deterministic application policy even if several agents
agree that it is appropriate.

Task-completion scorers can help stop iterative work when quality conditions are met, but deterministic
step limits, deadlines, and loop limits remain mandatory. Never use a model scorer as the sole stop
condition for spending or side effects.

## Communication Contracts

Use structured output for specialist results where possible:

```typescript
import { z } from 'zod'

const researchResultSchema = z.object({
  evidence: z.array(z.object({
    sourceId: z.string(),
    passage: z.string().max(2000),
    supports: z.string().max(500),
  })).max(5),
  insufficientEvidence: z.boolean(),
})
```

Validate IDs and source authorization again before using the evidence. A typed subagent result proves
shape, not provenance or truth. Define an explicit failure result instead of allowing a specialist to
return an ambiguous apology that another agent interprets as success.

## When Delegation Helps

Good reasons:

- Independent tools or permission boundaries
- Different models with separately evaluated quality/cost tradeoffs
- Context isolation for a large specialist task
- Parallel read-only research with a deterministic synthesis gate
- Separate evaluation datasets and release ownership

Poor reasons:

- A single prompt and one tool already meet requirements
- Agents merely pass prose back and forth without new information
- Splitting to avoid defining the actual completion contract
- Simulating a debate as a substitute for ground truth or tests

## Evaluate the System and Each Specialist

Test each subagent independently against its contract, then test supervisor delegation: correct
specialist selection, no unnecessary delegation, bounded depth/steps, filtered context, cancellation,
approval, failure propagation, and final evidence preservation. Include cases where a specialist is
wrong or injected and the supervisor must not amplify it.

Trace parent and nested spans to attribute latency, tokens, tool failures, and repeated delegation.
Optimize the measured bottleneck rather than adding more agents.

## Delegation Review Checklist

- [ ] Each agent has a distinct responsibility, minimal tools, and independent evals.
- [ ] Descriptions and instructions define when to delegate and when to stop.
- [ ] Supervisor and specialist steps, iterations, deadlines, and cost are bounded.
- [ ] Forwarded context is minimal; saved memory is not assumed to transfer.
- [ ] Tenant, authorization, and approval scope cannot broaden through delegation.
- [ ] Cancellation and failures propagate to nested work.
- [ ] Typed outputs are revalidated and untrusted summaries cannot trigger effects alone.

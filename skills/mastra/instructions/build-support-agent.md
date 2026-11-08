# Build a Mastra Support and Refund Agent

Create exactly these two artifacts in the current workspace:

- `src/mastra/index.ts`
- `design.md`

Write a complete, readable TypeScript module for a customer-support agent that can look up an
order and request a refund. Target the current stable **`@mastra/core` 1.82.0** APIs. This is an
offline design task: do not install packages, compile, run the agent, call a model, or make network
requests. An injected application service may represent order and payment side effects; do not add
a payment SDK or credentials.

## Required TypeScript behavior

1. Import `Agent` from `@mastra/core/agent`, `createTool` from `@mastra/core/tools`, and
   `createWorkflow`/`createStep` from `@mastra/core/workflows`. Use Zod schemas. Configure the
   agent with a current `provider/model` string such as `openai/gpt-6.1-sol` and realistic support
   instructions. Pass `structuredOutput: { schema: ... }` backed by a Zod object to `.generate()`.
2. Provide at least an order-lookup tool and a refund tool with `createTool`. Each tool must have
   validated `inputSchema` and `outputSchema`, plus an `execute: async (inputData, context) => ...`
   implementation. Do not put side effects in schema validation.
3. Mark the sensitive refund tool `requireApproval: true`. Derive actor and tenant identity only
   from trusted, server-populated `context.requestContext`; never authorize from model/client
   fields in `inputData`. Perform an application-service authorization check and pass a stable,
   tenant-scoped idempotency key to the refund service so retries and approval replays are safe.
4. Define a typed workflow with `createWorkflow` and `createStep`, including Zod input/output
   schemas. Compose it with `.then(step).commit()`. Demonstrate `await workflow.createRun()` and
   `await run.start({ inputData, ... })`; discriminate `result.status` (success, failure, and
   suspension) before accessing status-specific fields.
5. Bound agent execution with a small explicit `maxSteps` value (no more than 8).
6. Configure `Memory` from `@mastra/memory` with persistent `LibSQLStore` from `@mastra/libsql`.
   `Memory` requires `storage`; do not use `:memory:`. When generating, pass stable trusted
   `memory: { resource, thread }` identifiers derived from server request context, with the tenant
   in the resource scope.

## Required design explanation

In `design.md`, explain concrete failure and recovery behavior for all of the following:

- how approval is bound to the exact refund arguments and trusted identity, including changed
  arguments and approval replay;
- tenant authorization and why model/client-provided identity is never trusted;
- idempotency-key construction, atomic service enforcement, and replay behavior;
- workflow snapshot persistence and recovery after process restart, including approval suspension;
- the execution bound and what happens when it is exhausted;
- treatment of tool results and RAG/retrieved text as untrusted data, including prompt injection;
- offline evaluation and tracing, including useful metadata that excludes secrets and sensitive
  message contents.

Choose reasonable names and internal structure; no single exact implementation is required. Do not
use pseudocode, placeholders, TODOs, or claims that live integration was tested.

# Memory and Context Engineering

Memory stores conversation state; retrieval supplies shared knowledge; workflow storage preserves
execution snapshots. Keep these concerns separate. Mastra memory examples below use `@mastra/memory`
and `@mastra/libsql`, checked on 2026-11-08.

## Configure Persistent Message Memory

```bash
npm install @mastra/core @mastra/memory @mastra/libsql zod
```

```typescript
import { Agent } from '@mastra/core/agent'
import { LibSQLStore } from '@mastra/libsql'
import { Memory } from '@mastra/memory'

const storage = new LibSQLStore({
  id: 'support-storage',
  url: process.env.DATABASE_URL ?? 'file:./data/mastra.db',
})

const supportAgent = new Agent({
  id: 'support-agent',
  name: 'Support assistant',
  model: 'openai/gpt-6.1-sol',
  instructions: 'Answer from the current ticket and relevant conversation. Ask when context is missing.',
  memory: new Memory({
    storage,
    options: { lastMessages: 20 },
  }),
})
```

`Memory` needs storage for message history. `:memory:` is useful for tests, not durable production
state. Do not place mutable database files in a public assets directory. Configure retention,
backup, encryption, and deletion according to the data classification.

## Scope Every Call

```typescript
await supportAgent.generate(input, {
  memory: {
    resource: authenticatedUser.id,
    thread: ownedConversation.id,
  },
})
```

A thread's owning resource cannot later be changed. Derive IDs from authenticated server state and
verify that the conversation belongs to that resource before calling. Never accept arbitrary resource
or thread IDs from model arguments or untrusted client JSON.

Mastra also supports scope forms with `resource:` and `thread:` scopes. Choose the form supported by
the installed version, but preserve the same server-enforced ownership rule. A stable resource groups
a user's threads; a thread identifies one conversation.

## Choose a Memory Feature by Need

| Need | Feature or pattern | Caution |
| --- | --- | --- |
| Recent conversation | `lastMessages` | More history increases noise and cost |
| Durable user preference | Working memory with a typed schema | Validate and authorize updates |
| Related older messages | Semantic recall with vector store/embedder | Similarity is not permission or truth |
| Long-running conversation | Observational Memory | Requires storage and shutdown draining |
| Sensitive system context | Memory processors or filtering | Test redaction and information loss |
| Shared product documents | Retrieval pipeline | Not conversation memory |
| Interrupted workflow state | Persistent workflow snapshots | Not a chat history substitute |

Working memory is appropriate for a small set of stable, user-approved facts. Do not accumulate secrets,
full documents, access tokens, or speculative inferences. Expose correction and deletion paths.

## Long Conversations

Observational Memory can replace raw history with compact observations for long conversations. In that
mode, `lastMessages` does not provide the usual raw-history window. It requires storage, and asynchronous
buffering must settle before shutdown:

```typescript
await memory.settled()
```

Call this during application graceful shutdown when the configured memory supports it. Also test what
happens when summarization fails, the process stops abruptly, or observations become stale. Keep critical
business facts in typed application records rather than relying on a summary to preserve them.

## Semantic Recall

Semantic recall requires a vector store and embedder in addition to message storage. Query within the
already authorized thread/resource scope. Embedding text can reveal sensitive information to the
embedding provider, so apply the same privacy and data-residency review as for generation models.

Recall is a relevance mechanism, not proof. Include timestamps and source context where needed, and do
not let an old remembered preference override an explicit current instruction without confirmation.

## Context Budget

Build model context in priority order:

1. Stable operating instructions and output contract.
2. Current user request and authenticated task state.
3. Minimum relevant conversation memory.
4. Authorized retrieved evidence with provenance.
5. Bounded tool results needed for the next action.

Drop low-value tool payloads and stale details before increasing context size. Long context does not
guarantee use of the right fact and can increase injection surface, latency, and cost. Measure quality
against real maximum-size conversations.

## Subagent and Delegation Context

Delegation creates separate subagent execution context. Passing a summary to a subagent is not the same
as sharing the parent's saved memory. Forward only necessary fields and use message filtering when
parent history contains sensitive or irrelevant material. A subagent must not inherit broader tenant
scope or credentials.

## Failure and Privacy Cases

Test:

- A user attempts to open another user's thread.
- A model supplies a different resource ID.
- Old history conflicts with current authoritative account state.
- Retrieved memory contains a prompt injection.
- Working memory contains a fact the user deleted.
- The process stops with buffered observations pending.
- Storage is unavailable or the embedding provider fails.

Memory failures should degrade according to the task: a read-only recommendation may ask to retry,
while a payment flow must continue to rely on authoritative transactional state, never remembered state.

## Memory Review Checklist

- [ ] Storage is configured and suitable for the deployment lifecycle.
- [ ] Resource and thread ownership are derived and verified server-side.
- [ ] Conversation, knowledge, snapshots, and business records are separate.
- [ ] Retention, deletion, backup, and provider privacy are defined.
- [ ] Working memory is typed, minimal, correctable, and non-authoritative.
- [ ] Semantic recall cannot cross tenant or conversation boundaries.
- [ ] Long-conversation buffering is drained during shutdown.
- [ ] Context and evaluation cover the largest real conversation.

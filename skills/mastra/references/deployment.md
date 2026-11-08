# Deployment and Production Readiness

A Mastra application is production-ready only when its model dependencies, state, side effects, and
recovery behavior are operable. Build success alone does not prove agent correctness. Deployment APIs
checked on 2026-11-08.

## Build and Run the Mastra Server

```bash
mastra build
mastra start
# Or run the self-contained output directly:
node .mastra/output/index.mjs
```

`mastra build` produces `.mastra/output`. Keep build-time `bundler`, `deployer`, and `server` settings
as direct properties of the `new Mastra(...)` configuration. Verify exact deployer settings for the
chosen platform rather than copying another provider's configuration.

The server exposes `GET /health`. Add application readiness that checks required storage and critical
dependencies without making paid model calls on every probe. Liveness should not fail merely because a
transient optional provider is unavailable.

## Externalize State

Use durable shared stores for:

- Conversation memory and observational buffers
- Workflow snapshots and suspended runs
- Scorers and evaluation persistence
- Vector indexes and source metadata
- Application-owned approvals, operation IDs, receipts, and audit records

Do not put mutable database files under `src/mastra/public` or any public static directory. Local
`:memory:` and file databases are useful for development but do not provide shared multi-instance
recovery. Test backup restoration and retention deletion, not only connection success.

Keep secrets in platform secret storage. Pass only required environment variables to MCP subprocesses
and other child processes. Never expose provider keys through tool output, traces, client bundles,
error responses, or review screens.

## Streaming and Process Lifetime

Plain streams do not survive process replacement. During deploys, either drain active requests before
termination or explicitly tell clients to reconnect/restart through an application-level recovery path.
Durable workflows and suspended agents require persistent snapshots and rediscovery; they do not make
an already disconnected HTTP stream resumable by itself.

Graceful shutdown should:

1. Stop accepting new work.
2. Drain active requests within the configured timeout.
3. Stop queues and workers from claiming new operations.
4. Await pending memory buffering such as `memory.settled()` where configured.
5. Close request/shared MCP clients and storage connections.
6. Exit with an honest status when the drain deadline expires.

Use deployment drain settings compatible with the longest permitted request. Cancellation should reach
agents, tools, workflow steps, and provider calls through abort signals.

## Multi-Instance Coordination

Shared storage enables recovery but does not automatically prevent two workers from executing the same
business effect. Use queue visibility/leases, ownership checks, database constraints, and provider
idempotency as appropriate. Suspended-run resume needs authorization and stale/duplicate decision checks.

Avoid local in-memory rate limits or approval stores when several instances serve the same tenant. Keep
operation IDs and idempotency keys stable across instances and deployment versions.

## Release Checklist by Layer

| Layer | Release evidence |
| --- | --- |
| Contracts | Schema, API, tool, and workflow compatibility tests |
| Safety | Denied access, injection, approval tampering, and secret-redaction tests |
| Reliability | Timeout, retry, replay, restart, resume, and reconciliation tests |
| Quality | Representative deterministic gates and task-quality evals for every model candidate |
| Performance | p50/p95/p99 latency, throughput, context size, concurrency, and cost measurements |
| Operations | Health, traces, alerts, dashboards, runbooks, backup/restore, and rollback rehearsal |

Deploy model, prompt, tool, workflow, and retrieval changes as versioned behavior. Record the versions
in safe trace metadata so a quality regression can be attributed. Pin package versions and review
changelog/migration guidance before upgrades.

## Rollout and Rollback

Roll out read-only changes first when possible, then enable writes for a small authorized cohort. Monitor
denials, no-evidence outcomes, tool failures, human overrides, duplicate operations, latency, and cost,
not only model quality.

Rollback must account for suspended runs created by the newer workflow. Keep snapshot readers compatible
or migrate/finish them deliberately. Rolling back code does not undo external writes, embeddings, or
stored observations. Use feature flags to disable sensitive capabilities independently of deployment.

## Incident Controls

Provide controls to:

- Disable a tool, MCP server, model, or tenant without a full redeploy.
- Stop new workflow starts while preserving recovery data.
- Revoke credentials and disconnect affected request clients.
- Identify runs by safe trace/run IDs without exposing raw sensitive payloads.
- Reconcile unknown external outcomes using operation/idempotency IDs.
- Restore storage and verify business invariants before resuming writes.

Run incident drills for provider outage, prompt injection in a document, cross-tenant filter failure,
duplicate callback, stuck suspended run, and payment timeout. Assign human ownership for queued reviews.

## Production Review Checklist

- [ ] Build and direct output startup work with the deployed Node version.
- [ ] Health/readiness and graceful drain are tested under active work.
- [ ] All required state is durable, shared, protected, and restorable.
- [ ] Streams and suspended runs have explicit process-replacement behavior.
- [ ] Multi-instance writes use coordination and idempotency.
- [ ] Model/prompt/tool/workflow versions are attributable in traces.
- [ ] Rollout, capability kill switches, rollback, and reconciliation are rehearsed.
- [ ] No production validation depends solely on a happy-path model call.

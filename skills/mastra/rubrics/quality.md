# Mastra Support/Refund Agent Quality Rubric

Inspect only `src/mastra/index.ts` and `design.md` as text. Do not execute candidate code, install
packages, or make network requests. Score **0.0–1.0** using the weighted criteria below. The
deterministic grader checks the required API and safety surface; use this rubric for coherence,
specificity, and craftsmanship rather than awarding points for repeated keywords.

- **0.20 — API and type coherence:** The TypeScript plausibly composes current Mastra 1.82.0 agent,
  tool, workflow, memory, and Zod APIs. Schemas and result types describe meaningful support/refund
  data, and status-specific workflow fields are accessed safely.
- **0.25 — Security and financial safety:** Approval, trusted actor/tenant context, authorization,
  tenant isolation, and idempotency form one enforceable flow. Approval cannot be silently applied
  to changed refund arguments, and model or retrieved content cannot grant authority.
- **0.20 — Lifecycle and recovery:** Persistent memory/storage, stable resource/thread identity,
  workflow suspension, snapshots, restart recovery, bounded execution, and failure/suspension
  handling are internally consistent. The design does not equate conversation memory with a
  financial transaction ledger.
- **0.20 — Operational reasoning:** The design gives concrete replay, exhausted-step, untrusted
  tool/RAG input, evaluation, and tracing behavior. Proposed traces/evaluations are useful without
  claiming live validation or logging secrets and unnecessary personal data.
- **0.15 — Readability and scope:** The module is complete and maintainable, with clear names,
  focused service boundaries, useful instructions, and no dead code, pseudocode, TODOs, credential
  material, or unnecessary SDK/network integration.

A strong solution should score 0.9–1.0. Missing approval binding, tenant authorization, persistent
recovery, or idempotent side effects should keep the score below 0.5 even if the syntax looks
polished. Output a single number from 0.0 to 1.0 as the final score.

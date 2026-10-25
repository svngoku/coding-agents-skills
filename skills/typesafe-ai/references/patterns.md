# Patterns and Cookbook Map

Snapshot: 2026-10-25. Read the closest linked cookbook and current interface docs
before implementing. These are reusable compositions, not fixed recipes or
performance guarantees.

## Composition rules

- Exact retrieval, parsing, normalization, and execution remain in code.
- Batch independent questions sharing state, including explicit speculative branches.
- Use another request when earlier results change evidence or candidates.
- Keep raw judgments separate from policy so thresholds and weights can change.
- State injection or a model's approval never grants permission to execute.
- Track coverage, false positives/negatives, review load, latency, and cost.

Sources: [patterns](https://docs.typesafe.ai/patterns.md),
[intent routing](https://docs.typesafe.ai/patterns/intent-routing.md),
[fan-out](https://docs.typesafe.ai/patterns/fan-out.md),
[composite scoring](https://docs.typesafe.ai/patterns/composite-scoring.md).

## Route and fill closed-set arguments

1. [Function calling](https://docs.typesafe.ai/cookbooks/function_calling.md):
   choose a handler and ask its bounded argument questions speculatively.
   Consume only the selected handler's arguments. A minimum of confidences is
   not a joint correctness probability. Code validates and authorizes the call.
2. [Skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion.md):
   rank a catalog, fetch shortlisted candidates, then re-check them. This is a
   legitimate two-stage dependency. Its evaluation is synthetic; Choice and
   independent fit Nouls can disagree.

## Verify evidence and gate risk

3. [Guardrails](https://docs.typesafe.ai/cookbooks/llm_guardrails.md):
   separate hazard Nouls from severity Scores and deterministic policy precedence.
   Do not present the resulting filter as injection-proof.
4. [Citation checking](https://docs.typesafe.ai/cookbooks/citation_check.md):
   find exact spans in code, then judge support/contradiction. Literal matching
   can reject acceptable paraphrases unless deliberately adapted.
5. [RAG passage classification](https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md):
   independently judge relevance, evidence, contradiction, and injection.
   The page warns, "Nothing here is a security boundary."
6. [Extraction cascade](https://docs.typesafe.ai/cookbooks/sde_cascade.md):
   verify fields and escalate on an error. First validate required-field
   completeness in code: iterating only existing fields can miss omissions.
   In the published example, an empty object produces no per-field flags while
   the holistic head is excluded from that escalation loop.

## Retrieve, rank, and align

7. [Hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification.md):
   follow child Choices with beam search. Geometric-mean path numbers are ranking
   heuristics, not calibrated leaf correctness. Although prose mentions batching,
   the example issues separate concurrent requests for beam candidates.
8. [Reranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md):
   retrieve cheaply, then judge candidate/query pairs. Reranking cannot recover
   missing candidates. Its reported historical top-1 result is still only 18%;
   treat improvement against that baseline as context, not production readiness.
9. [Entity alignment](https://docs.typesafe.ai/cookbooks/entity_alignment.md):
   combine an ordered different/related/same Score with field-specific Nouls.
   Rounding the expectation embeds a policy. Retain distributions and evaluate
   false-merge consequences.
10. [Line search](https://docs.typesafe.ai/cookbooks/semantic_find.md):
    choose a source line ID and separately judge whether an answer exists.
    A forced Choice can be confident even when no line answers the question.

## Select source values and recover structure

11. [Pre-parsed values](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md):
    over-find candidates, select, copy verbatim, then normalize. Candidate recall,
    source offsets, and locale-aware parsing remain application responsibilities.
12. [Date extraction](https://docs.typesafe.ai/cookbooks/date_extraction_cookbook.md):
    select bounded components; assemble and validate dates in code. Calendar
    conventions are explicit policy. A minimum component confidence is not joint
    date correctness.
13. [Structure recovery](https://docs.typesafe.ai/cookbooks/autoformat.md):
    judge line stitching, then classify the resulting blocks. "Mid-sentence"
    is a more concrete boundary than vague "same paragraph." Published cost prose
    and output disagree; do not quote either as current pricing.

## Retain facts and respond to changing state

14. [Memory](https://docs.typesafe.ai/cookbooks/basic_memory.md):
    gate durable source facts and replay relevant context. Do not store inferred
    answers as newly observed facts. The demo does not establish bounded growth,
    expiry, persistence, or a complete privacy policy.
15. [Twitch filtering](https://docs.typesafe.ai/cookbooks/twitch_chat_filter.md):
    combine category and independent toxicity/spoiler judgments. Policy can change
    without recomputing unchanged judgments. Preserve IDs/order and specify whether
    pending items are visible before a result arrives.

## Abstain, measure, and learn

16. [Confidence classification](https://docs.typesafe.ai/cookbooks/classification_using_confidence.md):
    choose a specific label or coarsen to its parent. Its tail-sensitivity
    explanation conflicts with the current Choice confidence formula; prefer the
    dedicated confidence specification.
17. [Choice consistency](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook.md):
    illustrate abstention below a top-probability threshold. The published 0.60
    is an example. Repeated agreement is not accuracy, and changing a nonce
    confounds an identical-input repeatability measurement.
18. [Noul consistency](https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook.md):
    illustrate an inclusive 0.30–0.70 uncertainty band. It neither makes the model
    deterministic nor proves automatic decisions correct.
19. [Parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions.md):
    share a large document across questions. The reported 12.2x cost and 10x
    latency improvement belongs to that workload and a sequential baseline.
    Do not advertise it as a universal concurrency benchmark.
20. [Feature discovery](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery.md):
    propose semantic features, obtain Jev judgments, fit a supervised model, and
    iterate against development feedback. Keep final test labels out of discovery.
    One published run does not establish general model superiority. Clarify
    provider/legal restrictions before downstream-learning uses.

## Beyond the cookbooks

The [smart-home demo](https://docs.typesafe.ai/demos/smart-home.md) discusses
compound-request detection, generative splitting, and conversational fallback.
Its page promises source availability rather than supplying a full implementation.

The [use-case map](https://docs.typesafe.ai/concepts/use-case-map.md) is a discovery
aid, not a boundary on what can be built. Prefer composable judgments over a
generic "classify everything" prompt.

## Evidence discipline

Many cookbook results are cached historical runs, often using Jev 1.12.
Reading `json_cache.json`, reproducing a table, or reusing a threshold is not a
live service test or an application-specific accuracy evaluation.

Before adopting a pattern:

1. Establish candidate/source coverage and a deterministic baseline.
2. Define labels, no-match/absence cases, and consequential mistakes.
3. Separate mock, cached, and live results.
4. Evaluate held-out representative and adversarial cases.
5. Measure deployed behavior, not just the primitive's top answer.

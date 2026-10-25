# Official Sources and Research Ledger

Research date: **2026-10-25**.

The documentation pass enumerated **131 unique indexed URLs**: 130 documentation
pages and the external OpenAPI contract. Every entry below was fetched with
**HTTP 200 and reviewed**, including substantive cookbook code and generated SDK
reference pages. This records documentation coverage, not SDK execution, benchmark
reproduction, independent compliance verification, or production-code approval.

Start future work from [the live index](https://docs.typesafe.ai/llms.txt).
Read the relevant contract and closest cookbook rather than loading the entire
inventory for each request. Recheck model availability, limits, pricing, and SDK
types before relying on this snapshot.

## Additional first-party sources

These linked agreements/help pages informed the safety guidance:

- <https://typesafe.ai/legal/mca>
- <https://typesafe.ai/legal/data-processing>
- <https://typesafe.ai/legal/privacy-policy>
- <https://help.typesafe.ai/articles/5974925716-how-is-submitted-data-used-or-retained>

Linked repositories, datasets, videos, replay assets, and auth-gated compliance
reports were not exhaustively followed or independently verified.
The legacy documentation path `/migrating-to-v1.md` was probed separately, returned
404, and was absent from the current index. It is not included among the 131
successful indexed URLs.

## How to resolve drift

- Use the selected interface's schema/installed types for transport details.
- Use dedicated primitive/confidence specifications over contradictory cookbook prose.
- Label cached/historical examples; never promote them to live verification.
- Record unresolved differences explicitly and validate consequential behavior.
- Keep the original MIT license and upstream attribution when modifying this skill.
- Do not overwrite local changes by automatically reinstalling an upstream skill.

The operations reference records the concrete contradictions observed in this
pass. The indexed source inventory follows; every item has the status described
above.

## Indexed URLs

- <https://docs.typesafe.ai/introduction.md>
- <https://docs.typesafe.ai/introduction/quickstart.md>
- <https://docs.typesafe.ai/introduction/coding-agents.md>
- <https://docs.typesafe.ai/concepts/use-case-map.md>
- <https://docs.typesafe.ai/concepts/system-one.md>
- <https://docs.typesafe.ai/concepts/state.md>
- <https://docs.typesafe.ai/primitives.md>
- <https://docs.typesafe.ai/primitives/choice.md>
- <https://docs.typesafe.ai/primitives/score.md>
- <https://docs.typesafe.ai/primitives/noul.md>
- <https://docs.typesafe.ai/primitives/box.md>
- <https://docs.typesafe.ai/primitives/advanced.md>
- <https://docs.typesafe.ai/images.md>
- <https://docs.typesafe.ai/confidence.md>
- <https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md>
- <https://docs.typesafe.ai/introduction/machine-learning-primer.md>
- <https://docs.typesafe.ai/patterns.md>
- <https://docs.typesafe.ai/patterns/fan-out.md>
- <https://docs.typesafe.ai/patterns/confidence-routing.md>
- <https://docs.typesafe.ai/patterns/composite-scoring.md>
- <https://docs.typesafe.ai/patterns/intent-routing.md>
- <https://docs.typesafe.ai/cookbooks.md>
- <https://docs.typesafe.ai/cookbooks/function_calling.md>
- <https://docs.typesafe.ai/cookbooks/skill_suggestion.md>
- <https://docs.typesafe.ai/cookbooks/llm_guardrails.md>
- <https://docs.typesafe.ai/cookbooks/citation_check.md>
- <https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md>
- <https://docs.typesafe.ai/cookbooks/basic_memory.md>
- <https://docs.typesafe.ai/cookbooks/hierarchical_classification.md>
- <https://docs.typesafe.ai/cookbooks/classification_using_confidence.md>
- <https://docs.typesafe.ai/cookbooks/twitch_chat_filter.md>
- <https://docs.typesafe.ai/cookbooks/sde_cascade.md>
- <https://docs.typesafe.ai/cookbooks/date_extraction_cookbook.md>
- <https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md>
- <https://docs.typesafe.ai/cookbooks/autoformat.md>
- <https://docs.typesafe.ai/cookbooks/rerank_typesafe.md>
- <https://docs.typesafe.ai/cookbooks/semantic_find.md>
- <https://docs.typesafe.ai/cookbooks/entity_alignment.md>
- <https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook.md>
- <https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook.md>
- <https://docs.typesafe.ai/cookbooks/parallel_questions.md>
- <https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery.md>
- <https://docs.typesafe.ai/demos.md>
- <https://docs.typesafe.ai/demos/smart-home.md>
- <https://docs.typesafe.ai/models.md>
- <https://docs.typesafe.ai/agent-skill.md>
- <https://docs.typesafe.ai/legal.md>
- <https://docs.typesafe.ai/model-jaggedness/jev-1.13.md>
- <https://docs.typesafe.ai/api/system-one.md>
- <https://docs.typesafe.ai/api/list-models.md>
- <https://docs.typesafe.ai/api/errors.md>
- <https://docs.typesafe.ai/sdk.md>
- <https://docs.typesafe.ai/sdk/python.md>
- <https://docs.typesafe.ai/sdk/python/usage/basic.md>
- <https://docs.typesafe.ai/sdk/python/usage/advanced.md>
- <https://docs.typesafe.ai/sdk/python/changelog.md>
- <https://docs.typesafe.ai/sdk/python/api.md>
- <https://docs.typesafe.ai/sdk/python/api/clients/async.md>
- <https://docs.typesafe.ai/sdk/python/api/clients/sync.md>
- <https://docs.typesafe.ai/sdk/python/api/types/questions.md>
- <https://docs.typesafe.ai/sdk/python/api/types/assets.md>
- <https://docs.typesafe.ai/sdk/python/api/types/responses.md>
- <https://docs.typesafe.ai/sdk/python/api/retries.md>
- <https://docs.typesafe.ai/sdk/python/api/types/common.md>
- <https://docs.typesafe.ai/sdk/python/api/exceptions.md>
- <https://docs.typesafe.ai/sdk/python/api/constants.md>
- <https://docs.typesafe.ai/sdk/javascript.md>
- <https://docs.typesafe.ai/sdk/javascript/changelog.md>
- <https://docs.typesafe.ai/sdk/javascript/api.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/APIConnectionError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/APIError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/APIPromise.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/APITimeoutError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/APIUserAbortError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/AuthenticationError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/BadRequestError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/Image.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/InternalServerError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/NotFoundError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/PermissionDeniedError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/RateLimitError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/TypeSafeClient.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/TypeSafeError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/classes/UnprocessableEntityError.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/BoxCoordinates.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/BoxQuestion.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/BoxResponse.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/ChoiceQuestion.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/ChoiceResponse.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/Logger.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/ModelCard.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/Models.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/NoulQuestion.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/NoulResponse.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/Questions.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/RequestOptions.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/RetryPolicy.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/ScoreQuestion.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/ScoreResponse.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/SystemOneRequest.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/SystemOneResult.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/TextWithAssets.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/TypeSafeClientConfig.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/Usage.md>
- <https://docs.typesafe.ai/sdk/javascript/api/interfaces/WithResponse.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/Asset.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/ChoiceCriteria.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/Compression.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/ContentValue.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/Description.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/EntryType.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/EnvVar.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/Fetch.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/JsonValue.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/LogLevel.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/Question.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/ResultFor.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/ScoreCriteria.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/ScoreLegend.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/ScoreOf.md>
- <https://docs.typesafe.ai/sdk/javascript/api/type-aliases/SentContent.md>
- <https://docs.typesafe.ai/sdk/javascript/api/variables/ENV.md>
- <https://docs.typesafe.ai/sdk/javascript/api/variables/LOG_LEVELS.md>
- <https://docs.typesafe.ai/sdk/javascript/api/variables/noul.md>
- <https://docs.typesafe.ai/sdk/javascript/api/variables/score.md>
- <https://docs.typesafe.ai/sdk/javascript/api/variables/VERSION.md>
- <https://docs.typesafe.ai/sdk/javascript/api/variables/withAssets.md>
- <https://docs.typesafe.ai/sdk/javascript/api/functions/box.md>
- <https://docs.typesafe.ai/sdk/javascript/api/functions/choice.md>
- <https://docs.typesafe.ai/sdk/javascript/api/functions/image.md>
- <https://api.typesafe.ai/openapi.json>

Score the agent's solution 0.0-1.0:
- Did it follow the full lifecycle: ingest → adapt → wait → evaluate → export?
- Did it request a cost estimate before running the real job on a large dataset?
- Did it set an idempotency key for safe retries?
- Did it enable hallucination mitigation and at least one recipe (reasoning traces / deduplication)?
- Did it poll for completion with a timeout instead of assuming sync behavior?
- Did it fetch quality metrics and download the result via the returned URL?
- Is the code clean, typed where relevant, and runnable as a standalone script?
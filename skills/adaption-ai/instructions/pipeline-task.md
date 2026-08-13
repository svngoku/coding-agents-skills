Write a Python script (pipeline.py) that builds a complete Adaption dataset augmentation
pipeline using the adaption SDK (https://docs.adaptionlabs.ai/). The script must:
1. Create a client from the ADAPTION_API_KEY environment variable.
2. Upload a local CSV (data.csv with columns "instruction" and "response").
3. Run an adaptation job that: maps the prompt column to "instruction",
   enables hallucination mitigation (web-search grounding), requests
   reasoning traces + deduplication recipes, sets max_rows=500, and uses
   an idempotency key so retries are safe.
4. Wait for completion with a timeout, then fetch the quality evaluation
   and download the augmented dataset via the returned URL.
Use best practices: estimate the cost before starting the real run, and
poll rather than assuming synchronous completion.
Save the final script as pipeline.py in the current directory.
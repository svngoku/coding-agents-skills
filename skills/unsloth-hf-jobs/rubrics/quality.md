Score the agent's solution (the train.sh script) from 0.0 to 1.0. Read
train.sh and evaluate these four criteria:

- Flag correctness: are all required options present with the exact required
  values (--flavor a100-large, --secrets HF_TOKEN, --timeout 4h, --dataset,
  --output-repo, --lora-r 16, --learning-rate 2e-4, --batch-size 2,
  --gradient-accumulation 4, --eval-split 0.1, --merge-model,
  --trackio-space <username>/trackio), using the correct VLM script
  (sft-qwen3-vl.py) with job-level options before the "--" separator and
  script-level options after it?
- Smoke-test-first: is there a --max-steps 10 smoke-test variant placed before
  the full run, with a comment explaining why (verify setup cheaply before
  spending GPU hours; config errors cost money)?
- Completeness / merge-vs-adapter awareness: is --merge-model present, and do
  the comments/script show awareness that it uploads a merged, deployable
  model rather than just a LoRA adapter (adapter-only is fine for further
  fine-tuning, not for deployment)?
- Clarity of comments: do comments explain each section (smoke test first,
  secrets, trackio monitoring, merge model), and is the script clean,
  readable, and runnable (shebang, consistent line continuations)?

Assign 1.0 when all four are fully met; 0.7 when mostly correct with one
partial miss (e.g. --merge-model present but unexplained, or a smoke test
without a why-comment); 0.4 when several flags are wrong/missing or the wrong
script is used; 0.0 when no train.sh was produced or the script is
fundamentally wrong.

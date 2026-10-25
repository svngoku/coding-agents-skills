# Support-Triage Design Quality

Score 0–1 based on the submitted `request.json` and `design.md`.

- **Question meaning (0.25):** Instructions identify the evidence and full judgment.
  Questions do not depend on one another. Choice's `other` covers no-match and
  insufficient evidence. Noul concerns billing rather than permission or intensity.
- **Score interpretation (0.20):** Levels run routine to severe, are independently
  descriptive, and are not treated as exact counts or probabilities of permission.
- **Response and policy boundaries (0.25):** The design validates actual response
  contracts, preserves uncertainty/distributions, and keeps authorization and
  refund execution outside model judgments.
- **Evaluation discipline (0.20):** Thresholds are proposals requiring held-out
  evaluation against error costs, including absence and ambiguity, not universal
  safety guarantees.
- **Evidence and claims (0.10):** Synthetic text is untrusted evidence; no credential
  exposure or live calls are required. Offline checks are not presented as proof
  of SDK compatibility, server acceptance, semantic accuracy, or readiness.

Do not award semantic credit merely because the JSON passes structural checks.

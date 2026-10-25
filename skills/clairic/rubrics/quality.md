# Invoice Extractor Quality

Score each criterion from 0 to 1, based on the actual `invoice.py`, `requirements.txt` and
`design.md`, not merely a claim that they are correct. Deterministic checks already gate the
typed schema, exact SDK call profile, import-time structure and synthetic acceptance outcomes.

1. **Accurate ownership and capability boundary.** The note explicitly distinguishes the
   unofficial `clairic==0.1.0` Python SDK from Cloudflare's hosted
   `@cf/cloudflare/clairic` service. It correctly explains text-only SDK requests and absent
   image/options/stream kwargs without claiming those service features are SDK features.
2. **Uncertainty and review policy.** Explain `p_null` as absence probability, confidence as
   conditional on a value existing, and why neither a confident guess nor truncation may
   become an authoritative amount. Label 0.5/0.9 as synthetic task thresholds, not universal
   calibrated safety guarantees. Real review/calibration and business checks are required.
3. **Operational honesty.** Account-scoped credentials remain server-side and outside source.
   Authorized live inference transmits data and is metered; repeated calls/retries can cost
   again, multiple field passes affect input usage, preview limits may change, and a 30-second
   SDK HTTPX timeout is not a separate end-to-end wall-clock deadline. No unsupported claims
   about automatic retry, unlimited free inference, privacy guarantees or released local weights.
4. **Computation and authority.** The solution clearly assigns arithmetic and business rules
   to ordinary code and independently reviewed policy. An accepted extraction must not directly
   trigger a payment or be described as authorization; approval and independent checks remain
   necessary.
5. **Evidence and maintainability.** The short, readable implementation matches the explanation.
   State that static/AST and synthetic helper tests are offline, not evidence of model accuracy,
   calibration, live permissions/quotas, latency or real billing. Propose those as separately
   authorized live checks and do not invent completed tests or production-readiness evidence.

A note consisting only of keywords, or contradicting the code, earns no credit for the relevant
criterion. Do not infer live execution from the presence of an uncalled live helper.

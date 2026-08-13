# Rubric: Secure Login Rewrite — score 0.0–1.0

Judge the agent's `secure_app.py` holistically. The deterministic checks
verify the required mechanisms exist; this rubric scores the *quality of the
security reasoning*. Score the four dimensions below (0.0–0.25 each) and sum,
or judge holistically against the anchors. Anchor: 1.0 = code a security
reviewer would approve for production as-is.

## Threat-model awareness (0.0–0.25)
- The rewrite recognizes the specific flaws in the fixture and addresses
  them: SQL injection (parameterized/ORM queries), user enumeration
  (identical generic error, no timing/status-code oracle), broken
  authentication (argon2id/bcrypt, not plaintext), insecure secrets.
- No new attack surface introduced: no eval/exec/pickle on untrusted input,
  no debug/admin backdoors, no client-trusted authorization decisions.
- If the design uses sessions, threat-relevant choices are stated (why
  HttpOnly/Secure/SameSite, why a random 256-bit id).

## Defense-in-depth (0.0–0.25)
- Controls are layered, not single: hashing AND rate limiting AND generic
  errors AND env-managed secrets AND hardened cookies together.
- Timing-safe credential check for unknown users (dummy-hash verify or
  equivalent comparable work) so response time does not reveal existence.
- Session hygiene if sessions are used: rotation/expiry, server-side
  revocation on logout, idle/absolute timeouts.
- Failure path is handled: generic 401, no stack traces or DB errors leaked.

## Least privilege & minimal surface (0.0–0.25)
- Minimal endpoint surface: only the login endpoint (plus strictly
  necessary support); no unused routes, admin bypasses, or "test" helpers.
- Queries fetch only what the login needs; no whole-table dumps or
  unnecessary data in responses.
- Secrets are read from the environment with no hardcoded defaults; nothing
  sensitive is logged or returned.

## Code clarity & correctness (0.0–0.25)
- Valid, idiomatic Python that is runnable in structure (imports resolve,
  functions/endpoints are coherent, no syntax or logic gaps).
- Readable: sensible naming, short functions, comments explain *why*
  (security rationale) rather than restating the code.
- The solution reads as a deliberate, defensible hardening of the original
  rather than a copy-paste of the skill's worked example with no reasoning.

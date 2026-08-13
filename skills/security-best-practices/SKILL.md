---
name: security-best-practices
description: >
  Practical security review and hardening guidance for application code in
  Python, JavaScript/TypeScript, and Go. Use this skill whenever the user asks
  for a security review or audit, wants to harden authentication (password
  hashing, sessions, JWT, OAuth2/OIDC, MFA), fix OWASP Top 10 issues
  (injection, XSS, IDOR, CSRF, SSRF, deserialization), threat-model with
  STRIDE, clean up secrets and .env handling, or set up dependency scanning
  (pip-audit, npm audit, osv-scanner, gitleaks, SBOM). Also trigger for "make
  this secure", "is this safe to deploy", bcrypt/argon2id, rate limiting
  logins, RBAC/ABAC, CSP and security headers, cookie flags, and
  secure-by-default coding help.
---

# Security Best Practices

Review and harden application code against the vulnerabilities that actually get exploited: injection, broken authentication, IDOR, XSS, insecure secrets, weak dependencies. This skill is **actionable engineering guidance, not a compliance checklist** — each vulnerability class maps to a concrete code-level fix, with cheat-sheets for Python, JavaScript/TypeScript, and Go plus a worked secure-login example.

## Quick Reference

| Topic | Reference |
|-------|-----------|
| STRIDE worksheet, trust boundaries, least privilege, DFDs | [threat-modeling.md](references/threat-modeling.md) |
| JWT verification, OAuth2 flows, OIDC, token storage | [jwt-oauth.md](references/jwt-oauth.md) |

## Core Workflow

1. **Scope the review.** List entry points (HTTP routes, message consumers, CLIs, file processors), sensitive data (PII, credentials, payment data), and trust boundaries. Note the languages/frameworks so the right cheat-sheet applies.
2. **Threat-model in 5 minutes.** Run the STRIDE table below per entry point; mark high-risk flows (auth, uploads, deserialization, outbound HTTP). Full worksheet: [threat-modeling.md](references/threat-modeling.md).
3. **Walk the OWASP map.** Check every applicable row in the table below. Most findings come from three rows: access control (IDOR), injection, and security misconfiguration.
4. **Verify secure defaults and apply the cheat-sheet.** TLS, headers, CORS, cookies, non-leaky errors (Secure Defaults), then grep for your language's dangerous patterns (SQL f-strings, pickle, innerHTML, eval) and fix confirmed hits.
5. **Report findings.** Prioritize by severity × likelihood; give `file:line`, the vulnerability class, why it matters, and the fix. Offer to implement the fixes.

## Threat Modeling in 5 Minutes

| Letter | Threat | Ask | Example | Typical fix |
|--------|--------|-----|---------|-------------|
| S | Spoofing | Can someone pretend to be another user/service? | Forged JWT, stolen session | Strong auth, signed tokens, verify identity |
| T | Tampering | Can data be modified in transit or at rest? | SQL injection, unsigned payloads | TLS, signatures/HMAC, parameterized queries |
| R | Repudiation | Can an action be denied after the fact? | "I never made that transfer" | Audit logs, signed receipts |
| I | Information disclosure | Can sensitive data leak? | Stack traces, IDOR, secrets in logs | Least privilege, redaction, generic errors |
| D | Denial of service | Can the service be overwhelmed? | Expensive queries, no rate limits | Rate limiting, resource limits, timeouts |
| E | Elevation of privilege | Can a user do more than allowed? | Role escalation, IDOR | Server-side authorization on every action |

**Trust boundaries** sit wherever untrusted input crosses into trusted code (HTTP body → DB query, user input → shell, upload → storage, deserialized data → app, browser → API, app → internal services); every crossing is an attack surface. **Least privilege**: DB app-user gets `SELECT`/`INSERT` on its schema, never `DROP`/admin; run the OS process unprivileged; scoped IAM roles with short-lived credentials; no "admin mode" that bypasses checks.

## OWASP Top 10 → Code-Level Fixes

| Vulnerability | OWASP '21 | Code-level fix |
|---------------|-----------|----------------|
| Injection (SQL, NoSQL, command) | A03 | Parameterized queries/ORM bindings, no shell string interpolation |
| Broken authentication | A07 | argon2id/bcrypt, MFA, login rate limiting, session rotation |
| Sensitive data exposure | A02 | TLS in transit, encryption at rest, no plaintext secrets/PII, no hand-rolled crypto |
| Broken access control / IDOR | A01 | Server-side object-level checks; never trust client-supplied IDs or roles |
| Cross-site scripting (XSS) | A03 (injection) | Context-aware output encoding, CSP, no dangerous sinks |
| Insecure deserialization | A08 | Never `pickle`/`yaml.load` untrusted data; allowlist formats, signed payloads |
| Logging & monitoring failures | A09 | Log auth events, redact secrets/PII, alert on anomalies |
| Server-side request forgery (SSRF) | A10 | Allowlist destinations; block link-local and cloud-metadata IPs |
| Security misconfiguration | A05 | Secure defaults: headers, CORS, error messages, debug off |
| Vulnerable & outdated components | A06 | Lockfiles, `pip-audit`/`npm audit`/`osv-scanner` in CI, SBOM |

## Authentication

### Password storage

Never store plaintext, reversible encryption, or fast hashes (MD5/SHA-1/SHA-256 — trivially brute-forced). Use a dedicated password-hashing KDF:

```python
# Python (argon2id — preferred)
from argon2 import PasswordHasher
ph = PasswordHasher()                # defaults: m=65536, t=3, p=4
stored = ph.hash(password)           # store this string
ph.verify(stored, password)          # raises on mismatch
```

```javascript
import { hash, verify } from "@node-rs/argon2";   // or bcrypt with cost >= 12
const stored = await hash(password, { memoryCost: 65536, timeCost: 3, parallelism: 4 });
await verify(stored, password);
```

Salt is automatic with these libraries; bcrypt cost ≥ 12; rehash on upgrade from legacy MD5/SHA1 stores.

### Sessions, MFA, and rate limiting

- Session ID: 256-bit random (`secrets.token_urlsafe(32)` / `crypto.randomBytes(32)`), never derived from input.
- Cookie: `HttpOnly`, `Secure`, `SameSite=Lax` (or `Strict`), `__Host-` prefix where supported.
- Rotate the session ID on login/privilege change; invalidate server-side on logout; idle timeout (~30 min) + absolute timeout (~24 h).
- TOTP via `pyotp` (Python) / `otplib` (Node), mandatory for admin accounts; rate limit login per IP **and** per account with backoff.
- Identical error messages and comparable work whether the user exists or not (no user enumeration, no timing side-channel).

## Authorization

| Model | What it checks | Use when |
|-------|----------------|----------|
| RBAC | Role → permission mapping (`admin`, `editor`) | Coarse-grained, stable roles |
| ABAC | Attribute/policy evaluation (user, resource, context) | Fine-grained, policy-driven access |
| Object-level (ACL/ownership) | The *resource* belongs to the *caller* | Per-record access — always |

**IDOR prevention**: never trust a client-supplied ID to imply access. Scope every query to the authenticated user and return 404 (not 403) when the record isn't theirs, so you don't create an ID-existence oracle:

```python
def get_invoice(invoice_id: int, user=Depends(get_current_user)):
    invoice = db.scalar(select(Invoice).where(Invoice.id == invoice_id,
                                              Invoice.owner_id == user.id))
    if invoice is None:
        raise HTTPException(404)      # same as "not found"
    return invoice
```

Enforce authorization at the service layer, not just the router, on **every** endpoint — including hidden admin routes, which must still be verified server-side.

## JWT, OAuth2, and OIDC

| Need | Use |
|------|-----|
| Web app with a backend and browser clients | Server-side sessions + `httpOnly` cookies (revocable, simplest) |
| Stateless API between your own services | Short-lived JWT (or opaque tokens from an API gateway) |
| Third-party login (Google, GitHub, Okta) | OAuth2/OIDC authorization code + PKCE |
| SPA / mobile with an external IdP | OIDC authorization code + PKCE; access token in memory, refresh token rotated via secure channel |
| Machine-to-machine | OAuth2 client credentials flow |

**JWT verification — non-negotiables** (full checklist + alg-confusion attack: [jwt-oauth.md](references/jwt-oauth.md)):

```python
import jwt
payload = jwt.decode(
    token,
    key=public_key,                    # from the issuer's JWKS — never hardcoded or ignored
    algorithms=["RS256"],              # allowlist — never "none" or HS256 with an RSA key
    audience="https://api.example.com",
    issuer="https://auth.example.com",
    options={"require": ["exp", "iat", "nbf"]},
)
```

- Verify **signature**, **exp**, **nbf**, **iss**, **aud** — always. Reject `alg: none`. JWTs are signed, **not encrypted** — never put secrets or PII in the payload.
- Access tokens short-lived (minutes); refresh tokens rotated and revocable server-side. Never store tokens in `localStorage` (any XSS reads them); first-party apps use `httpOnly` cookies; never log tokens or pass them in URLs.

## Input Validation & Injection

### SQL injection

Never build SQL with concatenation, f-strings, `%`, or `.format` — parameterize or use ORM bindings:

```python
# Bad — injectable via `name`
cursor.execute(f"SELECT * FROM users WHERE name = '{name}'")
# Good
cursor.execute("SELECT * FROM users WHERE name = %s", (name,))
```

With ORMs (SQLAlchemy, Prisma, Django) prefer the query-builder; raw SQL only with bound parameters. Beware string-concat into ORM "extra" clauses.

### XSS, CSRF, and SSRF

- Encode for the correct context (HTML, attribute, JS, URL). React/Vue auto-escape text; danger is in sinks: `innerHTML`, `document.write`, `dangerouslySetInnerHTML`, `v-html`. Add a restrictive CSP (`default-src 'self'`); sanitize rich HTML server-side (bleach, DOMPurify).
- `SameSite` cookies + anti-CSRF token for state-changing requests; validate `Origin`/`Referer` on non-cookie APIs. CORS is a read-policy, not a write-policy — never rely on it to stop CSRF.
- SSRF: scheme + host **allowlist**; forbid user-supplied URLs to internal services; block link-local (`169.254.169.254` metadata) and private ranges; disable redirects or re-validate each hop.

### File uploads & command injection

- Validate extension **and** MIME **and** magic bytes (they must agree); random server-side filenames outside the web root; size limits; never execute uploads.
- Never interpolate input into shell strings — pass args as a list, or `shlex.quote` if a shell is unavoidable:

```python
# Bad
os.system(f"ffmpeg -i {filename} out.mp4")
# Good
subprocess.run(["ffmpeg", "-i", filename, "out.mp4"], check=True)
```

## Secrets Management

- Never hardcode secrets or commit real `.env` values — commit `.env.example` (placeholders only); `.env*` in `.gitignore`.
- Load from environment variables; use a secrets manager beyond trivial cases (AWS Secrets Manager/KMS, Vault, Doppler, 1Password).
- Scan for leaks in CI (`gitleaks detect` / `trufflehog`; GitHub/GitLab secret scanning); rotate on a schedule and **immediately** on leak; prefer short-lived, auto-rotating credentials.
- Keep secrets out of logs, errors, and stack traces.

## Supply Chain Security

- **Commit lockfiles** (`package-lock.json`, `yarn.lock`, `poetry.lock`, `uv.lock`, hashed `requirements.txt`) for reproducible builds; generate an SBOM with `syft`/`cyclonedx`; use Dependabot/Renovate for automated updates.
- Scan dependencies on every PR in CI: `pip-audit` (Python), `npm audit`/`pnpm audit` (Node), `osv-scanner` (all), `govulncheck` (Go) — gate on high/critical.
- Watch for typosquatting and recently-published malicious packages; verify name and maintainer before adding dependencies.

| Strategy | Use for | Risk |
|----------|---------|------|
| Exact pins (`==1.2.3`, `1.2.3`) | Applications, reproducible deploys | You own every upgrade |
| Ranges + lockfile (`^1.2`, `~1.2`) | Libraries, fast-moving deps | Lockfile drift, supply-chain exposure |
| Floating (`*`, no pin) | Never | Non-reproducible, poisoned-by-upstream |

## Secure Defaults

- **TLS everywhere**: enforce HTTPS redirect + HSTS; never ship a disabled cert "for local testing".

| Header | Value | Prevents |
|--------|-------|----------|
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | SSL stripping, downgrade |
| `Content-Security-Policy` | `default-src 'self'` | XSS |
| `X-Content-Type-Options` | `nosniff` | MIME sniffing |
| `X-Frame-Options` (or CSP `frame-ancestors`) | `DENY` / `'none'` | Clickjacking |
| `Referrer-Policy` | `no-referrer` (or `strict-origin-when-cross-origin`) | Referrer leakage |
| `Permissions-Policy` | minimal set | Feature abuse (camera, geolocation) |

- **CORS**: reflect only allowlisted origins; never `Access-Control-Allow-Origin: *` with `Allow-Credentials: true`; minimal `Allow-Methods`. **Cookies**: `HttpOnly; Secure; SameSite=Lax` minimum; `__Host-` prefix for session cookies.
- **Errors**: log full stack traces server-side; return generic messages (`"Invalid credentials"`, `"Internal server error"`). Never expose DB errors, paths, framework versions, or "user not found" vs "wrong password". Debug off in prod (`DEBUG=False`, `NODE_ENV=production`).

## Language Cheat-Sheets

| Language | Pitfall | Why it's dangerous | Fix |
|----------|---------|--------------------|-----|
| Python | `pickle` / `yaml.load` on untrusted data | Arbitrary code execution | `yaml.safe_load`, JSON/allowlisted formats |
| Python | `eval()` / `exec()` | Code injection | Parse with `ast` or a real parser |
| Python | SQL via f-strings / `%` / `.format` | SQL injection | Parameterized queries / ORM bindings |
| Python | MD5/SHA-1 for passwords | Instant brute-force | `argon2-cffi` / `bcrypt` with cost params |
| Python | `requests.get(user_url)` unchecked | SSRF | Scheme/host allowlist, block private IPs |
| JS/TS | `innerHTML`, `document.write`, `dangerouslySetInnerHTML`, `v-html` | XSS | Auto-escaping, sanitize with DOMPurify, CSP |
| JS/TS | Untrusted deep-merge (`Object.assign`, `lodash.merge`) | Prototype pollution → RCE | Filter `__proto__`/`constructor`, `structuredClone` |
| JS/TS | `fetch(user_url)` / axios with user URL | SSRF | Host allowlist, block metadata IPs, validate redirects |
| JS/TS | `eval` / `new Function` | Code injection | Avoid; `JSON.parse` for data |
| JS/TS | JWT `alg: none` / missing verification | Forged tokens | Allowlist algs, verify exp/aud/iss |
| Go | `fmt.Sprintf` into SQL | SQL injection | `database/sql` placeholders: `db.Query("... WHERE id = ?", id)` |
| Go | Returning `err.Error()` / stack traces in HTTP responses | Information disclosure | Log internally, return generic 500 |
| Go | `math/rand` for tokens/IDs | Predictable values | `crypto/rand.Read` |
| Go | Default `http.Server` (no timeouts), `InsecureSkipVerify: true` | Slowloris, MITM | `ReadTimeout`/`WriteTimeout`/`IdleTimeout`; never skip TLS verify |
| Go | No `gosec` / `govulncheck` in CI | Known vulnerabilities ship | Add to CI, fix or document findings |

## Worked Example: Secure Login (FastAPI)

argon2id hashing, login rate limiting, timing-safe credential check without user enumeration, random sessions, hardened cookies:

```python
import secrets
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import FastAPI, HTTPException, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address

ph = PasswordHasher()                   # argon2id, OWASP params
DUMMY_HASH = ph.hash("not-a-real-password")   # timing parity for unknown users
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@app.post("/login")
@limiter.limit("5/minute")              # per-IP rate limit
async def login(request: Request, response: Response, username: str, password: str):
    user = db.get_user(username)
    stored = user.password_hash if user else DUMMY_HASH
    try:
        ph.verify(stored, password)
        valid = user is not None
    except VerifyMismatchError:
        valid = False
    if not valid:
        raise HTTPException(401, detail="Invalid credentials")  # generic

    session_id = secrets.token_urlsafe(32)    # 256-bit random session id
    db.create_session(session_id, user.id,
                      expires_at=datetime.now(timezone.utc) + timedelta(hours=24))
    response.set_cookie("session", session_id,
                        httponly=True, secure=True, samesite="lax", max_age=86_400)
    return {"ok": True}
```

The unknown-user branch verifies a dummy hash so timing doesn't reveal valid usernames; the 401 body is identical for "no such user" and "wrong password". The cookie is `HttpOnly` (XSS-immune reads), `Secure` (HTTPS only), `SameSite=Lax` (CSRF-resistant), and the session row is revocable server-side on logout. For MFA, require a TOTP code from `pyotp` for admin accounts before issuing the session.

## Anti-Patterns to Avoid

- **Rolling your own crypto** — use audited libraries; never invent algorithms, MACs, or password schemes.
- **Storing passwords with fast hashes or reversible encryption** — always argon2id/bcrypt.
- **Committing secrets** — real values in code, `.env`, or test fixtures are leaks, not config.
- **Trusting client-supplied IDs, roles, or `isAdmin` flags** — IDOR and privilege escalation by construction.
- **`eval`, `pickle`, or `yaml.load` on untrusted input** — code execution on demand.
- **Skipping JWT verification** (alg, exp, aud) — forged or expired tokens are as good as no auth.
- **Returning stack traces / DB errors to users** — information disclosure via the error path.
- **Treating this skill as a tick-box checklist** — prioritize by what attackers can reach; re-review when code changes.

## When to Use / Not Use

**Use this skill when:**
- Reviewing new or existing Python, JS/TS, or Go code for security issues.
- Writing secure-by-default auth, sessions, or APIs (login flows, JWTs, OAuth2/OIDC).
- Fixing specific vulnerability classes: injection, XSS, IDOR, CSRF, SSRF, insecure deserialization, secrets in code.
- Setting up dependency scanning, lockfiles, SBOMs, or secret scanning in CI.
- Preparing code for a pen test, security audit, or production release ("is this safe to deploy?").

**Not for:**
- **Compliance-only work** (SOC 2/ISO 27001 evidence collection, policy documents) — this skill is about code, not certifications.
- **Infrastructure-only security** (network firewalls, Kubernetes hardening, cloud IAM architecture) — use a platform-specific skill.
- **Designing novel cryptography** — no skill replaces a cryptographer for new algorithms/protocols.
- **General code review with no security angle** — use a general code-review skill to avoid noise.
- **Malware analysis / reverse engineering** — a different tooling and threat model entirely.

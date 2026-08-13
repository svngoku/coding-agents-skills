# Threat Modeling: STRIDE, Trust Boundaries, and Least Privilege

Threat modeling is a structured way to find vulnerabilities before attackers do: enumerate what an attacker can reach, ask the STRIDE questions, and record the fix that closes each gap. A ten-minute model of a feature beats an hour of ad-hoc "look for bugs" review. This file is the companion to the Threat Modeling section of the main SKILL.md — read it when you need the full worksheet, DFD guidance, or a worked example.

## When to threat-model

- Before writing a new feature or endpoint that handles input, money, or identity.
- When authentication or authorization changes (new roles, new session scheme, OAuth flow).
- Before a penetration test, security audit, or production release.
- After an incident or a major dependency upgrade.

Keep models lightweight and updated as code changes — a stale model is worse than none.

## The 5-minute STRIDE worksheet

For each entry point, ask one question per letter:

| Letter | Threat | Question to ask | Example finding | Typical fix |
|--------|--------|-----------------|-----------------|-------------|
| S | Spoofing | Can an actor pretend to be someone else? | Forged JWT (`alg: none`), session fixation | Verify signatures/identity, rotate session IDs |
| T | Tampering | Can data be modified without detection? | SQL injection, unsigned config | Parameterized queries, HMAC signatures, TLS |
| R | Repudiation | Can an action be denied after the fact? | No audit trail for admin actions | Append-only audit log, signed receipts |
| I | Information disclosure | Can sensitive data leak? | IDOR, stack traces in responses, secrets in logs | Object-level authorization, generic errors, redaction |
| D | Denial of service | Can the service be made unavailable? | Unbounded uploads, expensive queries | Rate limits, size limits, query caps |
| E | Elevation of privilege | Can a user exceed their permissions? | Client-sent role, IDOR | Server-side authorization on every action |

Score each finding (see Risk scoring below) and record it in a threat register with an owner.

## Drawing a data-flow diagram (DFD)

A DFD has four element types:

- **Process** — something that transforms data (your API, a worker).
- **Data store** — where data rests (DB, cache, object storage).
- **External entity** — outside your control (browser, third-party API, another team's service).
- **Trust boundary** — dashed line separating trusted from untrusted.

Minimal DFD for a login flow:

```text
 Browser (external) ──HTTP──▶ API process ──SQL──▶ Users DB (store)
                                │
                                └──token lookup──▶ Session store (store)

 Trust boundary: the HTTP edge — everything the browser sends is untrusted
```

Draw boundaries around every component you trust (your processes, your stores) and treat everything outside as attacker-controlled. Every arrow that crosses a boundary is an attack surface.

## Common trust boundaries

| Boundary | Why it matters | Attack example |
|----------|----------------|----------------|
| HTTP body → app parsing | First place untrusted bytes enter | Injection, deserialization bombs |
| User input → SQL query | Data mixed with code | SQL injection |
| User input → shell / command | Command built from input | Command injection |
| Upload → storage / renderer | Files interpreted later | Malicious file upload |
| Serialized data → app objects | Code execution on load | `pickle`, unsafe `yaml.load` |
| Browser → API | Ambient authority | CSRF, CORS misconfiguration |
| App → internal services | Pivot point | SSRF to metadata / k8s / databases |

## Least privilege in practice

- Processes run as unprivileged users; drop capabilities after binding privileged ports (80/443).
- Database roles: the app user gets only the CRUD it needs on its own schema; migrations run with a separate privileged role; no shared admin credentials.
- Cloud: per-service IAM roles, short-lived credentials (STS, workload identity); no long-lived root keys on dev machines.
- Secrets: each service can read only its own secrets; rotate and revoke on personnel change.
- Code: authorization checks live in the service layer; an "admin bypass" flag is a vulnerability, not a feature.

## Risk scoring (keep it simple)

Score Likelihood × Impact, each 1–3:

| Score | Action |
|-------|--------|
| 6–9 | Fix before release, or add a compensating control |
| 3–4 | Backlog with an owner and due date |
| 1–2 | Document and accept |

## Worked mini-model: POST /login

**DFD**: Browser (external) → API process → Users DB (store); API → Session store; API → TOTP service.

**STRIDE walkthrough**:

| Letter | Finding | Fix |
|--------|---------|-----|
| S | Session fixation; user enumeration via error text | Rotate session ID at login; identical error messages |
| T | Passwords stored with SHA-1 | argon2id with OWASP cost parameters |
| R | No login audit trail | Log auth events with IP/user-agent, redact secrets |
| I | Timing reveals valid usernames; stack traces in 500s | Dummy-hash verify path; generic errors |
| D | Brute force / credential stuffing | Rate limit per IP and per account with backoff |
| E | Client-sent role escalates to admin | Derive role server-side from the session |

The table plus fixes is the deliverable — a threat register the team can act on.

## Anti-patterns

- Modeling everything exhaustively on the first pass — start with the highest-value flows (auth, money, uploads) and iterate.
- A model that never changes — update it when code changes and after incidents.
- Findings without owners — every row needs a fix and an owner.
- Trusting client input "because it comes from our own frontend" — the frontend is an external entity.

# JWT, OAuth2, and OIDC Deep Dive

JWT (JSON Web Tokens), OAuth2, and OpenID Connect solve overlapping but different problems. This file is the companion to the JWT section of the main SKILL.md: when to use each, how to verify JWTs without the classic mistakes, and how to store tokens safely.

## JWT anatomy

A JWT is three base64url-encoded segments joined by dots: `header.payload.signature`.

```json
// header
{ "alg": "RS256", "typ": "JWT", "kid": "key-2024-01" }
// payload (claims)
{ "sub": "user-123", "iss": "https://auth.example.com", "aud": "https://api.example.com",
  "exp": 1735689600, "iat": 1735686000, "nbf": 1735686000, "jti": "abc123" }
```

- `sub` — subject: who the token is about.
- `iss` — issuer: must match the identity provider you expect.
- `aud` — audience: the token is only valid for the API named here.
- `exp` / `nbf` / `iat` — expiry, not-before, issued-at (Unix seconds).
- `jti` — unique token ID, useful for revocation lists.

JWTs are signed, **not encrypted** — anyone can read the payload. Never put secrets or sensitive PII inside.

## Sessions vs JWT

| Criteria | Server-side session | JWT |
|----------|--------------------|-----|
| Revocation | Instant (delete session row) | Hard — needs a denylist/revocation store |
| State | Stateful (session store) | Stateless (self-verifying) |
| Client storage | httpOnly cookie | Client-held token (memory / localStorage / header) |
| Best for | First-party web apps | Service-to-service, federated identity |

Rule of thumb: if you control the whole stack, server-side sessions are simpler and safer. Reach for JWT when a token must be verifiable by parties that don't share your session store.

## Verification checklist (non-negotiable)

1. Fetch the signing key from the issuer's **JWKS** endpoint (cache it; rotate via `kid`).
2. Verify the **signature** with an allowlisted algorithm (`RS256`, `ES256`) — never `none`, never HS256 with an RSA public key.
3. Verify `exp` (with leeway ≤ 60 s), `nbf`, and `iat` (reject future-issued tokens).
4. Verify `iss` equals your configured issuer.
5. Verify `aud` contains exactly your API's audience.
6. For high-value tokens, check `jti` against a revocation/denylist.

Python (PyJWT):

```python
import jwt
from jwt import PyJWKClient

jwks = PyJWKClient("https://auth.example.com/.well-known/jwks.json")
payload = jwt.decode(
    token,
    jwks.get_signing_key_from_jwt(token).key,
    algorithms=["RS256"],                    # allowlist
    audience="https://api.example.com",
    issuer="https://auth.example.com",
    options={"require": ["exp", "iat", "nbf"]},
)
```

Node (jose):

```javascript
import { jwtVerify, createRemoteJWKSet } from "jose";

const jwks = createRemoteJWKSet(new URL("https://auth.example.com/.well-known/jwks.json"));
const { payload } = await jwtVerify(token, jwks, {
  algorithms: ["RS256"],
  audience: "https://api.example.com",
  issuer: "https://auth.example.com",
});
```

## The alg-confusion attack

An attacker takes a valid RS256 token, rewrites the header to `alg: HS256`, and re-signs it using the **public** key as the HMAC secret — public keys are public. If the server looks up the verification key from the JWKS (which contains the public key) and accepts HS256, the forged token verifies. Related: `alg: none`, where older libraries accepted unsigned tokens. Both are fixed the same way: **allowlist the algorithms** and never infer the algorithm from the token. The libraries above reject both by default when `algorithms` is explicit.

## OAuth2 grant flows

| Flow | Used by | Notes |
|------|---------|-------|
| Authorization Code + PKCE | Web apps, SPAs, mobile | The default choice; PKCE binds the code to the client |
| Client Credentials | Machine-to-machine | No user involved; scoped client tokens |
| Device Authorization | CLIs, TVs, IoT | User authorizes on another device |
| Implicit (deprecated) | — | Token in URL fragment; CSRF/leakage-prone. Don't use. |
| ROPC (password grant) | — | Handing user credentials to a third party. Avoid. |

## OpenID Connect (OIDC)

OIDC adds an identity layer on top of OAuth2:

- **ID token** — a JWT with identity claims (`sub`, `name`, `email`, `email_verified`). Validate it like any JWT, plus check the `nonce` (binds the token to your login request; stops replay/CSRF).
- **Access token** — what your API actually checks; may be opaque rather than a JWT.
- **Scopes** — `openid profile email`: ask for the minimum.
- **userinfo endpoint** — fetch profile claims there instead of trusting ID-token claims (validate the token first).

## Token storage

| Location | Risk | Recommendation |
|----------|------|----------------|
| localStorage / sessionStorage | Readable by any XSS | Never store access/refresh tokens here |
| httpOnly Secure cookie | CSRF if SameSite missing | Sessions, and refresh tokens for web apps |
| Memory only (JS variable) | Lost on reload | SPA access tokens |
| Secure enclave / keychain | Platform-dependent | Mobile apps |

Refresh tokens live server-side when possible; what the client holds should be as short-lived and revocable as the flow allows.

## Refresh token handling

- Rotate on every use; if a rotated token is reused, revoke the whole token family (reuse detection).
- Short TTL; store server-side; revoke on logout, password change, or suspicious activity.
- Never in URLs, logs, or error messages.

## Common attacks and fixes

| Attack | Fix |
|--------|-----|
| CSRF on the OAuth callback | `state` parameter (plus PKCE) |
| Open redirect via `redirect_uri` | Exact allowlist of redirect URIs |
| Token in query string / logs | POST bodies and `Authorization` header only; never log tokens |
| Mix-up attack (swapped IdPs) | Pin the IdP per client; validate `iss` |
| Confused deputy / token substitution | Validate `aud` and scope on every API |

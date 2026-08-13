# Task: Harden a Vulnerable Login Endpoint

The workspace contains **vulnerable.py**, a login endpoint with several
critical security flaws (plaintext password storage, SQL injection, user
enumeration, no rate limiting, a hardcoded secret, and an unsafe "session"
token). Read it, identify every flaw, and rewrite it as a hardened login
endpoint.

**Save the result as `secure_app.py` in the current directory.** This is the
only file the grader inspects — there is no database server, no network, and
no test runner. Write plain Python (framework-agnostic) or FastAPI-style code.

## Requirements for `secure_app.py`

1. **Password hashing** — store and verify passwords with a dedicated
   password-hashing library: `argon2-cffi` (argon2id preferred, as in the
   skill's worked example), `bcrypt`, `passlib`, or
   `werkzeug.security`. Never compare plaintext, and never use fast hashes
   (MD5/SHA-1/SHA-256/hashlib) for passwords.
2. **No string-built SQL** — every database query must be parameterized
   (placeholders like `?` or `%s`) or go through an ORM/query-builder.
   Never build SQL with f-strings, `%` formatting, `.format()`, or string
   concatenation using user input.
3. **No user enumeration** — return the *same* generic error message (for
   example `"Invalid credentials"`) whether the username does not exist or
   the password is wrong. Do not leak "user not found" vs "wrong password",
   and do not answer with different HTTP status codes (404 vs 401) for the
   two cases.
4. **Login rate limiting** — limit failed login attempts per IP or via an
   in-memory counter (e.g. `slowapi`/Flask-Limiter, or a hand-rolled
   per-IP attempt counter with a time window).
5. **Secrets from the environment** — load secrets (DB path/credentials,
   signing keys) from `os.environ`/`os.getenv`; do not hardcode any
   secret strings in the source.
6. **Safe session handling** — if you set a session cookie, set it with
   `HttpOnly`, `Secure`, and `SameSite` flags, and use a random,
   unpredictable session id (e.g. `secrets.token_urlsafe(32)`). If your
   design has no sessions, that is acceptable.
7. **Minimal endpoint surface** — expose only the login endpoint (plus
   anything strictly needed to support it). No backdoors, debug routes, or
   admin bypasses.

## Quality bar

The solution should be the kind of code you would merge for a production
login: correct, readable, and defensible against the OWASP issues the
original file demonstrates (injection, broken authentication, information
disclosure, denial of service, insecure secrets).

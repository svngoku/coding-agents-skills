"""vulnerable.py - INSECURE login endpoint (do not deploy).

This is the starting point for a security rewrite exercise. It deliberately
contains several critical flaws. Read it carefully, identify every flaw, then
write a hardened replacement as secure_app.py.
"""

import sqlite3

# FLAW 1: hardcoded secret (and no secrets management at all)
SECRET_KEY = "s3cr3t-k3y-d0-n0t-l34k-2024"
DB_PATH = "app.db"

# FLAW 2: passwords stored and compared in plaintext (no hashing)
USERS = {
    "alice": "password123",
    "bob": "letmein",
}


def get_db():
    return sqlite3.connect(DB_PATH)


def login(username, password):
    conn = get_db()
    cur = conn.cursor()

    # FLAW 3: SQL injection - the query is built with an f-string and
    # interpolates untrusted user input directly into the statement.
    cur.execute(f"SELECT password FROM users WHERE username = '{username}'")
    row = cur.fetchone()

    if row is None:
        # FLAW 4: user enumeration - attackers can tell which usernames exist
        return {"error": "User not found"}

    if row[0] != password:
        # FLAW 4 (continued): a distinct message for the wrong password
        return {"error": "Wrong password"}

    # FLAW 5: no rate limiting - credentials can be brute-forced freely
    # FLAW 6: the "session token" is the hardcoded secret; no expiry, no
    #         cookie flags (HttpOnly / Secure / SameSite), no revocation
    return {"ok": True, "token": SECRET_KEY}

"""Storage for NutriFit AI: accounts, saved items and profiles.

Uses Postgres when DATABASE_URL is set (in Streamlit secrets or the environment),
otherwise a local SQLite file. Both keep the same tables and the same functions."""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

DB_PATH = os.environ.get("NUTRIFIT_DB", "nutrifit.db")
MAX_FAILS = 5        # wrong PINs before a temporary lock
LOCK_MINUTES = 15


def _database_url():
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    try:
        import streamlit as st
        return st.secrets.get("DATABASE_URL")
    except Exception:
        return None


PG_URL = _database_url()
PG = bool(PG_URL)


@contextmanager
def _connection():
    """One raw connection; commits on success, rolls back on error."""
    if PG:
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(PG_URL, row_factory=dict_row) as conn:
            yield conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()


_ready = False


def _ensure():
    """Create the tables once per process, the first time the database is used."""
    global _ready
    if not _ready:
        _create_tables()
        _ready = True


@contextmanager
def _db():
    _ensure()
    with _connection() as conn:
        yield conn


def _run(conn, sql, params=()):
    """Run a query. If a table is missing, create the tables and try once more."""
    sql = sql.replace("?", "%s") if PG else sql
    try:
        return conn.execute(sql, params)
    except Exception as e:
        if type(e).__name__ != "UndefinedTable" and "no such table" not in str(e):
            raise
        if PG:
            conn.rollback()
        _create_tables()
        return conn.execute(sql, params)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _hash(pin, salt):
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, 200_000)


def _create_tables():
    pk, idt, blob = ("BIGSERIAL PRIMARY KEY", "BIGINT", "BYTEA") if PG else ("INTEGER PRIMARY KEY", "INTEGER", "BLOB")
    unique_name = "" if PG else " UNIQUE COLLATE NOCASE"
    with _connection() as conn:
        conn.execute(f"""CREATE TABLE IF NOT EXISTS users (
            id {pk}, name TEXT NOT NULL{unique_name}, salt {blob} NOT NULL, pin_hash {blob} NOT NULL,
            created TEXT NOT NULL, failed INTEGER NOT NULL DEFAULT 0, locked_until TEXT)""")
        conn.execute(f"""CREATE TABLE IF NOT EXISTS items (
            id {pk}, user_id {idt} NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            kind TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL, meta TEXT,
            created TEXT NOT NULL, updated TEXT NOT NULL)""")
        conn.execute(f"""CREATE TABLE IF NOT EXISTS profiles (
            user_id {idt} PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            data TEXT NOT NULL, updated TEXT NOT NULL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS items_by_user ON items (user_id, kind, updated)")
        if PG:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS users_name_lower ON users (lower(name))")
        else:  # upgrade a database made by an earlier version
            have = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
            if "failed" not in have:
                conn.execute("ALTER TABLE users ADD COLUMN failed INTEGER NOT NULL DEFAULT 0")
            if "locked_until" not in have:
                conn.execute("ALTER TABLE users ADD COLUMN locked_until TEXT")


def init():
    """Create the tables now (they are also created automatically on first use)."""
    global _ready
    _create_tables()
    _ready = True


# ---------- Accounts ----------
def create_user(name, pin):
    """Returns (True, user) or (False, message)."""
    name = (name or "").strip()
    if not 2 <= len(name) <= 30:
        return False, "Choose a name between 2 and 30 characters."
    if not re.fullmatch(r"\d{6,8}", pin or ""):
        return False, "Your PIN must be 6 to 8 digits."
    salt = secrets.token_bytes(16)
    try:
        with _db() as conn:
            if _run(conn, "SELECT 1 FROM users WHERE lower(name) = lower(?)", (name,)).fetchone():
                return False, "That name is already taken. Try another."
            row = _run(conn, "INSERT INTO users (name, salt, pin_hash, created) VALUES (?, ?, ?, ?) "
                             "RETURNING id", (name, salt, _hash(pin, salt), _now())).fetchone()
            return True, {"id": row["id"], "name": name}
    except Exception:
        return False, "That name may already be taken. Try another."


def verify_user(name, pin):
    """Returns (user, None) on success, or (None, message). Locks briefly after repeated failures."""
    wrong = (None, "That name and PIN don't match. Check them and try again.")
    now = datetime.now(timezone.utc)
    with _db() as conn:
        row = _run(conn, "SELECT * FROM users WHERE lower(name) = lower(?)",
                   ((name or "").strip(),)).fetchone()
        if row is None or not pin:
            return wrong
        until = datetime.fromisoformat(row["locked_until"]) if row["locked_until"] else None
        if until and until > now:
            mins = int((until - now).total_seconds() // 60) + 1
            return None, f"Too many wrong tries. Try again in about {mins} minute(s)."
        if hmac.compare_digest(_hash(pin, bytes(row["salt"])), bytes(row["pin_hash"])):
            if row["failed"] or until:
                _run(conn, "UPDATE users SET failed = 0, locked_until = NULL WHERE id = ?", (row["id"],))
            return {"id": row["id"], "name": row["name"]}, None
        failed = row["failed"] + 1
        if failed >= MAX_FAILS:
            lock = (now + timedelta(minutes=LOCK_MINUTES)).isoformat()
            _run(conn, "UPDATE users SET failed = 0, locked_until = ? WHERE id = ?", (lock, row["id"]))
            return None, f"Too many wrong tries. Try again in {LOCK_MINUTES} minutes."
        _run(conn, "UPDATE users SET failed = ? WHERE id = ?", (failed, row["id"]))
        return wrong


def delete_account(user_id):
    with _db() as conn:
        _run(conn, "DELETE FROM users WHERE id = ?", (user_id,))


# ---------- Saved items (kind: chat, scan, plan, allergy) ----------
def save_item(user_id, kind, title, body, meta=None, item_id=None):
    """Update item_id if it belongs to this user, otherwise insert. Returns the item id."""
    meta_json = json.dumps(meta) if meta is not None else None
    now = _now()
    with _db() as conn:
        if item_id:
            cur = _run(conn, "UPDATE items SET title = ?, body = ?, meta = ?, updated = ? "
                             "WHERE id = ? AND user_id = ?", (title, body, meta_json, now, item_id, user_id))
            if cur.rowcount:
                return item_id
        row = _run(conn, "INSERT INTO items (user_id, kind, title, body, meta, created, updated) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
                   (user_id, kind, title, body, meta_json, now, now)).fetchone()
        return row["id"]


def list_items(user_id, kind, limit=50):
    with _db() as conn:
        rows = _run(conn, "SELECT id, title, body, meta, updated FROM items "
                          "WHERE user_id = ? AND kind = ? ORDER BY updated DESC, id DESC LIMIT ?",
                    (user_id, kind, limit)).fetchall()
    return [dict(r) for r in rows]


def counts(user_id):
    with _db() as conn:
        rows = _run(conn, "SELECT kind, COUNT(*) AS n FROM items WHERE user_id = ? GROUP BY kind",
                    (user_id,)).fetchall()
    return {r["kind"]: r["n"] for r in rows}


def delete_item(user_id, item_id):
    with _db() as conn:
        _run(conn, "DELETE FROM items WHERE id = ? AND user_id = ?", (item_id, user_id))


def delete_all(user_id):
    with _db() as conn:
        _run(conn, "DELETE FROM items WHERE user_id = ?", (user_id,))


# ---------- Profile ----------
def get_profile(user_id):
    with _db() as conn:
        row = _run(conn, "SELECT data FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
    return json.loads(row["data"]) if row else None


def save_profile(user_id, data):
    with _db() as conn:
        _run(conn, "INSERT INTO profiles (user_id, data, updated) VALUES (?, ?, ?) "
                   "ON CONFLICT(user_id) DO UPDATE SET data = excluded.data, updated = excluded.updated",
             (user_id, json.dumps(data), _now()))


def clear_profile(user_id):
    with _db() as conn:
        _run(conn, "DELETE FROM profiles WHERE user_id = ?", (user_id,))

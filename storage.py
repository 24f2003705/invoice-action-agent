
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("tasks.db")


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                principal TEXT NOT NULL,
                context_id TEXT NOT NULL,
                batch_id TEXT,
                state TEXT NOT NULL,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (principal, message_id),
                FOREIGN KEY (task_id) REFERENCES tasks(task_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_claims (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                task_json TEXT,
                PRIMARY KEY (principal, message_id)
            )
        """)



def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )


def get_task(task_id, principal):
    with connect() as conn:
        row = conn.execute(
            "SELECT task_json FROM tasks "
            "WHERE task_id = ? AND principal = ?",
            (task_id, principal)
        ).fetchone()

    return json.loads(row["task_json"]) if row else None


def list_tasks(principal):
    with connect() as conn:
        rows = conn.execute(
            "SELECT task_json FROM tasks "
            "WHERE principal = ? ORDER BY created_at DESC",
            (principal,)
        ).fetchall()

    return [json.loads(row["task_json"]) for row in rows]


def find_message(principal, message_id):
    with connect() as conn:
        row = conn.execute(
            """
            SELECT message_hash, task_id
            FROM messages
            WHERE principal = ? AND message_id = ?
            """,
            (principal, message_id)
        ).fetchone()

        if row is None:
            return None

        task_row = conn.execute(
            "SELECT task_json FROM tasks WHERE task_id = ?",
            (row["task_id"],)
        ).fetchone()

        if task_row is None:
            return None

        return {
            "message_hash": row["message_hash"],
            "task": json.loads(task_row["task_json"])
        }


def save_new_task(task, principal, message_id, message_hash, batch_id=None):
    init_db()

    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")

        existing = conn.execute(
            "SELECT message_hash, task_id FROM messages "
            "WHERE principal = ? AND message_id = ?",
            (principal, message_id)
        ).fetchone()

        if existing:
            if existing["message_hash"] != message_hash:
                raise ValueError("IDEMPOTENCY_CONFLICT")

            row = conn.execute(
                "SELECT task_json FROM tasks WHERE task_id = ?",
                (existing["task_id"],)
            ).fetchone()

            return json.loads(row["task_json"])

        conn.execute(
            """
            INSERT INTO tasks
                (task_id, principal, context_id, batch_id, state, task_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task["id"],
                principal,
                task["contextId"],
                batch_id,
                task["status"]["state"],
                canonical_json(task)
            )
        )

        conn.execute(
            """
            INSERT INTO messages
                (principal, message_id, message_hash, task_id)
            VALUES (?, ?, ?, ?)
            """,
            (principal, message_id, message_hash, task["id"])
        )

    return task


def update_task(task, principal):
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute(
            "SELECT state FROM tasks WHERE task_id = ? AND principal = ?",
            (task["id"], principal)
        ).fetchone()

        if row is None:
            raise LookupError("Task not found")

        old_state = row["state"]
        new_state = task["status"]["state"]

        if old_state in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED",
                         "TASK_STATE_REJECTED"):
            raise ValueError("Terminal tasks cannot be modified")

        conn.execute(
            """
            UPDATE tasks
            SET state = ?, task_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE task_id = ? AND principal = ?
            """,
            (new_state, canonical_json(task), task["id"], principal)
        )


def reserve_message(principal, message_id, message_hash):
    init_db()

    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute(
            """
            SELECT message_hash, status, task_json
            FROM message_claims
            WHERE principal = ? AND message_id = ?
            """,
            (principal, message_id)
        ).fetchone()

        if row:
            if row["message_hash"] != message_hash:
                return {"status": "conflict"}

            if row["status"] == "completed":
                return {
                    "status": "completed",
                    "task": json.loads(row["task_json"])
                }

            return {"status": "processing"}

        # Support tasks created before this table was added.
        old = conn.execute(
            """
            SELECT m.message_hash, t.task_json
            FROM messages m
            JOIN tasks t ON t.task_id = m.task_id
            WHERE m.principal = ? AND m.message_id = ?
            """,
            (principal, message_id)
        ).fetchone()

        if old:
            if old["message_hash"] != message_hash:
                return {"status": "conflict"}

            task_json = old["task_json"]
            conn.execute(
                """
                INSERT INTO message_claims
                    (principal, message_id, message_hash, status, task_json)
                VALUES (?, ?, ?, 'completed', ?)
                """,
                (principal, message_id, message_hash, task_json)
            )
            return {
                "status": "completed",
                "task": json.loads(task_json)
            }

        conn.execute(
            """
            INSERT INTO message_claims
                (principal, message_id, message_hash, status)
            VALUES (?, ?, ?, 'processing')
            """,
            (principal, message_id, message_hash)
        )

        return {"status": "new"}


def complete_message_claim(principal, message_id, message_hash, task):
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")

        cursor = conn.execute(
            """
            UPDATE message_claims
            SET status = 'completed', task_json = ?
            WHERE principal = ? AND message_id = ?
              AND message_hash = ? AND status = 'processing'
            """,
            (
                canonical_json(task),
                principal,
                message_id,
                message_hash
            )
        )

        if cursor.rowcount != 1:
            raise ValueError("Message reservation was lost")


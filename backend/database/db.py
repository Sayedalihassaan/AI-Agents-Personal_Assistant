import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Try importing PostgreSQL dependencies.
# If they're missing or the env-var is blank,
# the module falls back to an in-process dict.
# ──────────────────────────────────────────────
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

try:
    from config import POSTGRES_URL
except ImportError:
    POSTGRES_URL = ""

# In-memory fallback store  {user_id: [messages]}
_IN_MEMORY_STORE: Dict[str, List[Dict]] = {}
_USE_DB = False  # resolved at first connection attempt


def _can_use_db() -> bool:
    """Return True only when psycopg2 is installed AND a URL is configured."""
    return _PSYCOPG2_AVAILABLE and bool(POSTGRES_URL and POSTGRES_URL.strip())


def get_connection():
    """Return a psycopg2 connection or raise clearly."""
    return psycopg2.connect(POSTGRES_URL, cursor_factory=RealDictCursor)


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────
def InitializeChatHistoryTable():
    if not _can_use_db():
        logger.warning("⚠️  No POSTGRES_URL — using in-memory chat history (non-persistent).")
        return

    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id           SERIAL PRIMARY KEY,
                user_id      VARCHAR(255) NOT NULL UNIQUE,
                chat_history JSONB        NOT NULL,
                created_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info("💾 Chat history table initialized (PostgreSQL)")
    except Exception as e:
        logger.error(f"PostgreSQL init failed: {e} — falling back to in-memory store.")


# ──────────────────────────────────────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────────────────────────────────────
def load_chat_history(user_id: str) -> List[Dict[str, str]]:
    if not _can_use_db():
        return list(_IN_MEMORY_STORE.get(user_id, []))

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chat_history FROM chat_history WHERE user_id=%s LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            return row["chat_history"] if row else []
    except Exception as e:
        logger.error(f"load_chat_history DB error: {e} — using in-memory fallback.")
        return list(_IN_MEMORY_STORE.get(user_id, []))
    finally:
        if conn:
            conn.close()


def save_chat_history(user_id: str, chat_history: List[Dict[str, str]]):
    # Always keep in-memory copy (useful as fallback)
    _IN_MEMORY_STORE[user_id] = list(chat_history)

    if not _can_use_db():
        return

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_history (user_id, chat_history)
                VALUES (%s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET chat_history = EXCLUDED.chat_history;
                """,
                (user_id, Json(chat_history)),
            )
            conn.commit()
        logger.info(f"💾 Chat history saved to PostgreSQL for user={user_id}")
    except Exception as e:
        logger.error(f"save_chat_history DB error: {e} — saved to in-memory only.")
    finally:
        if conn:
            conn.close()


def delete_chat_history(user_id: str = None):
    """Delete history for one user (or all if user_id is None)."""
    if user_id:
        _IN_MEMORY_STORE.pop(user_id, None)
    else:
        _IN_MEMORY_STORE.clear()

    if not _can_use_db():
        return {"message": f"Chat history cleared (in-memory) for user={user_id}"}

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            if user_id:
                cur.execute("DELETE FROM chat_history WHERE user_id=%s", (user_id,))
            else:
                cur.execute("DELETE FROM chat_history")
        conn.commit()
        logger.info(f"🗑️  Chat history cleared for user={user_id}")
        return {"message": f"Chat history cleared for user={user_id}"}
    except Exception as e:
        logger.error(f"delete_chat_history DB error: {e}")
        return {"message": f"Cleared in-memory; DB error: {e}"}
    finally:
        if conn:
            conn.close()


def _delete_database():
    """Drop the chat_history table (dev utility)."""
    if not _can_use_db():
        _IN_MEMORY_STORE.clear()
        return
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS chat_history")
        conn.commit()
        cur.close()
        conn.close()
        logger.info("💾 Chat history table dropped")
    except Exception as e:
        logger.error(f"Failed to drop table: {e}")


def storage_mode() -> str:
    """Return a human-readable label for the current storage backend."""
    return "PostgreSQL" if _can_use_db() else "In-Memory (no DB)"
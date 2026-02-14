"""
streamlit_app.py  ——  AI Personal Assistant
Run:  streamlit run streamlit_app.py
"""
import asyncio
import os
import sys
import time
import traceback
import streamlit as st
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv(override=True)

# ── MUST be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="AI Personal Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Import diagnostics — capture EXACTLY which module failed
# ─────────────────────────────────────────────────────────────
_errors: Dict[str, str] = {}

try:
    from backend.services.agent_service.agent import response_to_json
except Exception as e:
    _errors["agent"] = f"{type(e).__name__}: {e}"
    response_to_json = None  # type: ignore

try:
    from backend.database.db import (
        load_chat_history, delete_chat_history,
        storage_mode, InitializeChatHistoryTable,
    )
    _DB_OK = True
except Exception as e:
    _errors["db"] = f"{type(e).__name__}: {e}"
    _DB_OK = False
    def load_chat_history(user_id): return []          # type: ignore
    def delete_chat_history(user_id=None): pass        # type: ignore
    def storage_mode(): return "Unavailable"           # type: ignore
    def InitializeChatHistoryTable(): pass             # type: ignore

_AGENT_OK = response_to_json is not None

# ─────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; }

/* sidebar */
[data-testid="stSidebar"]        { background:#0f172a !important; }
[data-testid="stSidebar"] *      { color:#e2e8f0 !important; }
[data-testid="stSidebar"] input  { background:#1e293b !important; border:1px solid #334155 !important; color:#f1f5f9 !important; }
[data-testid="stSidebar"] .stSelectbox > div { background:#1e293b !important; }

/* chat bubbles */
.chat-wrap { display:flex; flex-direction:column; gap:16px; padding:8px 0; }
.chat-row  { display:flex; align-items:flex-end; gap:10px; }
.chat-row.user      { flex-direction:row-reverse; }
.avatar { width:36px;height:36px;border-radius:50%;display:flex;align-items:center;
          justify-content:center;font-size:17px;flex-shrink:0; }
.avatar.user      { background:#6366f1; }
.avatar.assistant { background:#1e293b;border:1px solid #334155; }
.bubble { max-width:72%;padding:12px 16px;border-radius:18px;
          font-size:.94rem;line-height:1.6;word-break:break-word; }
.bubble.user      { background:#6366f1;color:#fff;border-bottom-right-radius:4px; }
.bubble.assistant { background:#f8fafc;color:#1e293b;border:1px solid #e2e8f0;
                    border-bottom-left-radius:4px; }
.msg-meta  { font-size:.68rem;color:#94a3b8;margin-top:3px; }
.msg-meta.user { text-align:right; }

/* status badges */
.badge { display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
         border-radius:99px;font-size:.72rem;font-weight:600;letter-spacing:.03em; }
.badge.ok   { background:#dcfce7;color:#166534; }
.badge.warn { background:#fef9c3;color:#854d0e; }
.badge.err  { background:#fee2e2;color:#991b1b; }

/* diagnostic box */
.diag-box { background:#1e1e1e;border:1px solid #ef4444;border-radius:10px;
            padding:14px 16px;font-family:monospace;font-size:.8rem;
            color:#fca5a5;white-space:pre-wrap;overflow-x:auto; }

/* input */
.stTextArea textarea { border-radius:12px !important;border:1px solid #cbd5e1 !important;
                        font-size:.94rem !important;resize:none !important; }
/* button */
div[data-testid="stButton"] button {
    background:#6366f1;color:#fff;border-radius:10px;
    padding:10px 26px;font-weight:600;border:none;
    width:100%;transition:background .2s;
}
div[data-testid="stButton"] button:hover { background:#4f46e5; }

.thin-hr { border:none;border-top:1px solid #e2e8f0;margin:10px 0; }
.section-label { font-size:.7rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:.08em;color:#64748b;margin:16px 0 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def run_async(coro):
    """Run async coroutine safely from Streamlit's sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def badge(ok: bool, on_text: str, off_text: str, warn=False) -> str:
    cls = ("ok" if ok else "warn") if (ok or warn) else "err"
    dot = "●"
    return f'<span class="badge {cls}">{dot} {on_text if ok else off_text}</span>'


def fmt_time(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def api_key_ok() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def openrouter_key() -> bool:
    k = os.getenv("OPENAI_API_KEY", "")
    return k.startswith("sk-or-v1") or k.startswith("sk-")


# ─────────────────────────────────────────────────────────────
# One-time DB init
# ─────────────────────────────────────────────────────────────
if "db_init_done" not in st.session_state:
    InitializeChatHistoryTable()
    st.session_state.db_init_done = True

# ─────────────────────────────────────────────────────────────
# Session defaults
# ─────────────────────────────────────────────────────────────
defaults = {
    "messages":       [],
    "timestamps":     [],
    "user_id":        "default_user",
    "model":          "openai/gpt-4o-mini",
    "history_loaded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Assistant")
    st.markdown('<hr class="thin-hr">', unsafe_allow_html=True)

    # User settings
    st.markdown('<p class="section-label">User Settings</p>', unsafe_allow_html=True)
    new_uid = st.text_input("User ID", value=st.session_state.user_id,
                            help="Unique ID — each user keeps separate history.")
    if new_uid != st.session_state.user_id:
        st.session_state.user_id        = new_uid
        st.session_state.messages       = []
        st.session_state.timestamps     = []
        st.session_state.history_loaded = False
        st.rerun()

    # Model selection
    st.markdown('<p class="section-label">Model (OpenRouter)</p>', unsafe_allow_html=True)
    model_options = [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-sonnet",
        "google/gemini-flash-1.5",
        "mistralai/mistral-7b-instruct",
    ]
    st.session_state.model = st.selectbox(
        "model", model_options, index=0, label_visibility="collapsed"
    )
    os.environ["AGENT_MODEL"] = st.session_state.model

    # Status indicators
    st.markdown('<p class="section-label">Status</p>', unsafe_allow_html=True)
    db_mode = storage_mode()
    db_ok   = "PostgreSQL" in db_mode
    _api_ok = api_key_ok()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(badge(_api_ok, "API ✓", "API ✗"), unsafe_allow_html=True)
        st.caption("OpenRouter key")
    with c2:
        st.markdown(badge(db_ok, "DB ✓", "RAM ⚡", warn=True), unsafe_allow_html=True)
        st.caption(db_mode)

    # Active tools
    st.markdown('<p class="section-label">Active Tools</p>', unsafe_allow_html=True)
    for icon, name in [("🔍","DuckDuckGo Search"), ("📧","Gmail"), ("🕐","Date & Time")]:
        st.markdown(f"{icon} {name}")

    st.markdown('<hr class="thin-hr">', unsafe_allow_html=True)

    # Clear history
    st.markdown('<p class="section-label">Session</p>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages       = []
        st.session_state.timestamps     = []
        st.session_state.history_loaded = False
        delete_chat_history(st.session_state.user_id)
        st.success("History cleared!")
        st.rerun()

    st.caption(f"User: `{st.session_state.user_id}`")
    st.caption("LangChain · OpenRouter · Streamlit")


# ─────────────────────────────────────────────────────────────
# MAIN — Header
# ─────────────────────────────────────────────────────────────
st.markdown("## 🤖 AI Personal Assistant")
st.caption("Powered by LangChain + OpenRouter — Search the web, manage Gmail, track time.")
st.markdown('<hr class="thin-hr">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Diagnostics banner (shown only when there are errors)
# ─────────────────────────────────────────────────────────────
if _errors:
    st.error("**Some modules failed to import.** See diagnostic info below.", icon="🚨")

    for mod, err in _errors.items():
        with st.expander(f"❌ `{mod}` — {err.split(':')[0]}", expanded=True):
            st.markdown('<div class="diag-box">' + err + '</div>', unsafe_allow_html=True)

            # Actionable hints
            if "AgentExecutor" in err or "create_tool_calling_agent" in err:
                st.warning(
                    "**Fix:** Run one of these in your conda environment:\n\n"
                    "```bash\n"
                    "pip install -U langchain langchain-openai langchain-community\n"
                    "# OR if you use langchain-classic:\n"
                    "pip install langchain-classic\n"
                    "```",
                    icon="💡",
                )
            if "langchain_classic" in err:
                st.warning(
                    "**Fix:** Install the langchain-classic package:\n\n"
                    "```bash\npip install langchain-classic\n```\n\n"
                    "Or remove `langchain_classic` imports and use `langchain` directly.",
                    icon="💡",
                )
            if "openai" in err.lower() or "api_key" in err.lower():
                st.warning(
                    "**Fix:** Add your OpenRouter key to `.env`:\n\n"
                    "```\nOPENAI_API_KEY=sk-or-v1-...\n```",
                    icon="💡",
                )

if not _api_ok:
    st.warning(
        "**`OPENAI_API_KEY` is missing.** Add it to your `.env` file:\n\n"
        "`OPENAI_API_KEY=sk-or-v1-your-key-here`",
        icon="🔑",
    )

# ─────────────────────────────────────────────────────────────
# Load persisted history (once per session)
# ─────────────────────────────────────────────────────────────
if not st.session_state.history_loaded and _AGENT_OK:
    try:
        saved = load_chat_history(st.session_state.user_id)
        if saved and not st.session_state.messages:
            for m in saved:
                st.session_state.messages.append({"role": m["role"], "content": m["content"]})
                st.session_state.timestamps.append(time.time())
    except Exception:
        pass
    st.session_state.history_loaded = True

# ─────────────────────────────────────────────────────────────
# Chat display
# ─────────────────────────────────────────────────────────────
chat_area = st.container()
with chat_area:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px 40px;color:#94a3b8;">
            <div style="font-size:3.5rem;margin-bottom:14px;">👋</div>
            <p style="font-size:1.1rem;font-weight:600;color:#64748b;margin-bottom:8px;">
                How can I help you today?
            </p>
            <p style="font-size:.9rem;line-height:1.7;">
                Try: <em>"What's the latest news about AI?"</em><br>
                or: <em>"Check my recent emails"</em><br>
                or: <em>"What time is it in Tokyo?"</em>
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        bubble_html = '<div class="chat-wrap">'
        for i, msg in enumerate(st.session_state.messages):
            role    = msg["role"]
            content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
            ts_val  = st.session_state.timestamps[i] if i < len(st.session_state.timestamps) else time.time()
            ts_str  = fmt_time(ts_val)
            avatar  = "🧑" if role == "user" else "🤖"
            bubble_html += f"""
            <div class="chat-row {role}">
                <div class="avatar {role}">{avatar}</div>
                <div>
                    <div class="bubble {role}">{content}</div>
                    <div class="msg-meta {role}">{ts_str}</div>
                </div>
            </div>"""
        bubble_html += '</div>'
        st.markdown(bubble_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Input bar
# ─────────────────────────────────────────────────────────────
st.markdown('<hr class="thin-hr">', unsafe_allow_html=True)
input_col, btn_col = st.columns([5, 1])

with input_col:
    user_input = st.text_area(
        "Message",
        key="user_input_field",
        placeholder="Type your message… (Shift+Enter for a new line)",
        height=80,
        label_visibility="collapsed",
    )

with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    send = st.button("Send ➤", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Send handler
# ─────────────────────────────────────────────────────────────
if send:
    query = (user_input or "").strip()

    if not query:
        st.warning("Please type a message before sending.", icon="⚠️")
    elif not _AGENT_OK:
        st.error(
            "The agent failed to load. Fix the import errors shown above, then restart Streamlit.",
            icon="🚨",
        )
    elif not _api_ok:
        st.error("Add your `OPENAI_API_KEY` to `.env` before sending.", icon="🔑")
    else:
        # Immediately show user bubble
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.timestamps.append(time.time())

        with st.spinner("Thinking…"):
            try:
                result  = run_async(response_to_json(user_query=query, user_id=st.session_state.user_id))
                answer  = result.get("output", "❌ No response received.")
                err_str = result.get("error")

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.timestamps.append(time.time())

                if err_str:
                    with st.expander("⚠️ Technical details (click to expand)", expanded=False):
                        st.code(err_str, language="text")

            except Exception as exc:
                tb = traceback.format_exc()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "❌ An unexpected error occurred. Please try again.",
                })
                st.session_state.timestamps.append(time.time())
                with st.expander("⚠️ Technical details", expanded=False):
                    st.code(tb, language="text")

        st.rerun()

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;padding:16px 0 2px;color:#94a3b8;font-size:.72rem;">'
    'AI Personal Assistant · LangChain · OpenRouter · Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
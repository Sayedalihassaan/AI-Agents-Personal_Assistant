"""
datetime_tool.py  —  Place at: backend/services/tools/datetime_tool.py
"""
import pytz
from typing import Optional
from datetime import datetime

# ── Tool import — compatible with all LangChain versions ──────
try:
    from langchain_core.tools import Tool
except ImportError:
    try:
        from langchain.tools import Tool           # type: ignore
    except ImportError:
        from langchain_community.tools import Tool  # type: ignore


def get_current_datetime(timezone: Optional[str] = None) -> str:
    """Return current date/time, optionally in a given timezone."""
    now = datetime.now()
    if timezone:
        try:
            tz  = pytz.timezone(timezone)
            now = datetime.now(tz)
            return f"Current date and time in {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        except Exception as e:
            return (
                f"Error with timezone '{timezone}': {e}. "
                f"Showing local time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
            )
    return f"Current local date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def datetime_tool() -> Tool:
    """Return a Tool that reports the current date and time."""
    return Tool(
        name="current_datetime",
        description=(
            "Useful for getting the current date and time. "
            "Input can be empty or a timezone string (e.g. 'UTC', 'America/New_York', 'Africa/Cairo'). "
            "Always use this when the user asks about the current time or date."
        ),
        func=get_current_datetime,
    )
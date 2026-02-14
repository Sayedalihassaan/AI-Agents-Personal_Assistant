"""
gmail_tool.py  —  Place at: backend/services/tools/gmail_tool.py
"""
import logging
from typing import List

# ── Tool import — compatible with all LangChain versions ──────
try:
    from langchain_core.tools import BaseTool
    _Tool = BaseTool
except ImportError:
    _Tool = object  # type: ignore

# ── Config import — handle both package and flat layouts ──────
try:
    from backend.config import GMAIL_TOKEN_PATH
except ImportError:
    try:
        from config import GMAIL_TOKEN_PATH         # type: ignore
    except ImportError:
        import os
        GMAIL_TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH", ".gmail_token.json")

from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GMAIL_TOOLS_CACHE = None


def get_cached_gmail_tools() -> List:
    """Initialize Gmail tools once and cache them in memory."""
    global GMAIL_TOOLS_CACHE

    if GMAIL_TOOLS_CACHE is not None:
        return GMAIL_TOOLS_CACHE

    try:
        logger.info("🔐 Initializing Gmail Tools (one-time setup)")
        credentials = get_gmail_credentials(
            token_file=GMAIL_TOKEN_PATH,
            scopes=["https://mail.google.com/"],
        )
        api_resource = build_resource_service(credentials=credentials)
        toolkit       = GmailToolkit(api_resource=api_resource)

        GMAIL_TOOLS_CACHE = toolkit.get_tools()
        logger.info(f"✅ Gmail Tools cached ({len(GMAIL_TOOLS_CACHE)} tools)")
        return GMAIL_TOOLS_CACHE

    except Exception as e:
        logger.error(f"❌ Gmail Tools unavailable: {e}")
        GMAIL_TOOLS_CACHE = []
        return []
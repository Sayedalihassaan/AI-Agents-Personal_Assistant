"""
search_tool.py  —  Place at: backend/services/tools/search_tool.py
"""
import logging

# ── Tool import — compatible with all LangChain versions ──────
try:
    from langchain_core.tools import Tool
except ImportError:
    try:
        from langchain.tools import Tool           # type: ignore
    except ImportError:
        from langchain_community.tools import Tool  # type: ignore

from langchain_community.tools import DuckDuckGoSearchResults

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def duckduckgo_search_tool() -> Tool:
    """Return a DuckDuckGo web-search tool compatible with all LangChain versions."""
    try:
        logger.info("Initializing DuckDuckGo Search Tool")
        search = DuckDuckGoSearchResults(output_format="list")
        return Tool(
            name="duckduckgo_search",
            description="Search the web for recent information. Input should be a search query string.",
            func=search.invoke,
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize DuckDuckGo Search Tool: {e}")
        raise
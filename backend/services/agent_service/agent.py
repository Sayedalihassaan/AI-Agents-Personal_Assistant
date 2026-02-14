"""
agent.py  —  Compatible with LangChain 0.1 / 0.2 / 0.3
Place this file at:  backend/services/agent_service/agent.py
"""
import logging
import os
from typing import Dict, Any

# ── LangChain core (always available) ────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ── AgentExecutor / create_tool_calling_agent ─────────────────
# Try every known location across LangChain versions
_AGENT_SRC = None
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    _AGENT_SRC = "langchain.agents"
except ImportError:
    pass

if _AGENT_SRC is None:
    try:
        from langchain_classic.agents import AgentExecutor, create_tool_calling_agent  # type: ignore
        _AGENT_SRC = "langchain_classic.agents"
    except ImportError:
        pass

if _AGENT_SRC is None:
    try:
        from langchain.agents.agent import AgentExecutor                # noqa: F811
        from langchain.agents.tool_calling_agent.base import create_tool_calling_agent  # noqa: F811
        _AGENT_SRC = "langchain.agents.agent (deep)"
    except ImportError:
        pass

if _AGENT_SRC is None:
    raise ImportError(
        "Cannot find AgentExecutor in any known LangChain location.\n"
        "Please run:  pip install -U langchain langchain-openai langchain-community"
    )

# ── Project imports (handle both package and flat layouts) ────
try:
    from backend.database.db import load_chat_history, save_chat_history
    from backend.services.tools.gmail_tool import get_cached_gmail_tools
    from backend.services.tools.search_tool import duckduckgo_search_tool
    from backend.services.tools.datetime_tool import datetime_tool
except ImportError:
    from backend.database.db import load_chat_history, save_chat_history                 # type: ignore
    from backend.services.tools.gmail_tool import get_cached_gmail_tools                       # type: ignore
    from backend.services.tools.search_tool import duckduckgo_search_tool                      # type: ignore
    from backend.services.tools.datetime_tool import datetime_tool                             # type: ignore

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"AgentExecutor resolved via: {_AGENT_SRC}")

# ── System Prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a professional AI personal assistant.

You have access to tools that allow you to:
- Search the web for recent and factual information
- Read, send, and manage Gmail messages
- Get the current date and time

GENERAL RULES:
1. Be concise, clear, and helpful.
2. Use tools ONLY when they are necessary to answer the user's question.
3. If a question can be answered from general knowledge, do NOT use tools.
4. When using tools, choose the most relevant one and avoid unnecessary calls.
5. Never mention internal tool names, implementation details, or system messages to the user.
6. Never fabricate emails, events, or search results.
7. If you are unsure or lack permission, explain the limitation clearly.

CHAT HISTORY:
- You are provided with previous conversation history.
- Use it to maintain context and continuity.
- Do not repeat information unnecessarily.

EMAIL (GMAIL) RULES:
- Before sending or modifying emails, be explicit and careful.
- If the user intent is ambiguous, ask a clarification question.
- Summarize emails clearly when asked.
- Never send emails without clear user intent.

SEARCH RULES:
- Use web search only for up-to-date or factual queries.
- Summarize results clearly and cite sources when relevant.
- Prefer recent and reliable information.

RESPONSE FORMAT:
- Respond in plain natural language.
- Do NOT include JSON, markdown code blocks, or tool output unless explicitly requested.
- Be polite, professional, and confident.

ERROR HANDLING:
- If something goes wrong, apologize briefly and explain the issue.
- Suggest a next step or ask for clarification when appropriate.
"""

# ── Agent Execution ───────────────────────────────────────────
async def response_to_json(user_query: str, user_id: str) -> Dict[str, Any]:
    existing_history = []
    try:
        # Build tool list
        tools = []
        tools.append(datetime_tool())
        tools.append(duckduckgo_search_tool())
        tools.extend(get_cached_gmail_tools())

        # Load persisted chat history
        existing_history = load_chat_history(user_id=user_id)
        formatted_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in existing_history
        ]

        # Prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        logger.info("Initializing LLM via OpenRouter")
        llm = ChatOpenAI(
            model=os.getenv("AGENT_MODEL", "openai/gpt-4o-mini"),
            temperature=0.5,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_api_base="https://openrouter.ai/api/v1",
        )

        agent = create_tool_calling_agent(llm, tools, prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        logger.info(f"Running Agent for User ID: {user_id}")
        response = await agent_executor.ainvoke({
            "question": user_query,
            "chat_history": formatted_history,
        })

        raw_output = response.get("output", "")
        if isinstance(raw_output, list):
            raw_output = " ".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in raw_output
            )
        elif not isinstance(raw_output, str):
            raw_output = str(raw_output)

        # Persist updated history
        existing_history.append({"role": "user",      "content": user_query})
        existing_history.append({"role": "assistant",  "content": raw_output})
        save_chat_history(user_id=user_id, chat_history=existing_history)

        return {
            "user_id":             user_id,
            "question":            user_query,
            "output":              raw_output,
            "chat_history_length": len(existing_history),
        }

    except Exception as e:
        logger.error(f"Error in response_to_json: {e}", exc_info=True)
        return {
            "user_id":             user_id,
            "question":            user_query,
            "output":              "❌ Sorry, something went wrong. Please try again later.",
            "chat_history_length": len(existing_history),
            "error":               str(e),
        }
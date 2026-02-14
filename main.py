import logging
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv
from backend.config import API_TOKEN
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from backend.services.agent_service.agent import response_to_json
from fastapi.middleware.cors import CORSMiddleware
from backend.database.db import (
    InitializeChatHistoryTable, 
    delete_chat_history, 
    load_chat_history, 
    save_chat_history,
)

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------------------- Load Environment Variables --------------------
_ = load_dotenv(override=True)  

# -------------------- Lifespan Context --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    try:
        InitializeChatHistoryTable()
        logger.info("🚀 Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

    # IMPORTANT: yield control to FastAPI
    yield

    # --- Shutdown ---
    delete_chat_history()
    logger.info("🛑 Application shutdown")

# -------------------- FastAPI App --------------------
app = FastAPI(
    title="Personalized AI Agent",
    version="1.0.0",
    description="AI Agent As Personal Assistant with Database Connectivity & GMAIL & Calendar Tools",
    lifespan=lifespan,
    # redoc_url=None,
    # docs_url=None,
)

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# -------------------- Models --------------------
class QueryRequest(BaseModel):
    user_id: str
    question: str

class QueryResponse(BaseModel):
    user_id: str
    question: str
    output: str
    chat_history_length: int
    error: str | None = None

# -------------------- Routes --------------------
@app.get("/")
async def root():
    return {
        "message": "Welcome to AI Agent API 🤖",
        "endpoints": {
            "POST /query": "Ask a question to the AI Agent",
            "GET /chat_history": "Get chat history for a specific teacher and school",
            "DELETE /chat_history/clear": "Clear chat history for a specific teacher and school"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def ask_database(request: QueryRequest, token: str=None):
    """
    Ask a question to the AI Agent.

    :param request: The query request containing the user ID and question.
    :type request: QueryRequest
    :param token: Optional API token for authentication.
    :type token: str

    :raises HTTPException: 401 Unauthorized if token is invalid, 400 Bad Request if user ID or question is empty, 500 Internal Server Error if query execution fails.

    :return: A dictionary containing the user ID, question, output, chat history length, and error (if any).
    :rtype: QueryResponse
    """
    try:
        if token != API_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized.")
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        if not request.user_id.strip():
            raise HTTPException(status_code=400, detail="User ID cannot be empty.")

        # --- Reset chat history ---
        if request.question == "/reset":
            save_chat_history(request.user_id, [])
            return QueryResponse(
                user_id=request.user_id,
                question=request.question,
                output="🗑️ Chat history has been reset.",
                chat_history_length=0,
                error=None
            )

        # --- Execute query via AI SQL Agent ---
        result = await response_to_json(
            user_query=request.question,
            user_id=request.user_id,
        )
        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
## =================================================================================
## ==================== Chat History Management ============== ## ==================
## =================================================================================

@app.get("/chat_history")
async def get_chat_history(user_id: str, token: str = None):

    try:
        if token != API_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized.")
        
        history = load_chat_history(user_id=user_id)
        return {
            "user_id": user_id,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Failed to fetch chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat_history/clear")
async def clear_chat_history(user_id: str, token: str = None):
    """
    Clears the chat history for a specific user.

    :param user_id: The user ID to clear the chat history for.
    :type user_id: str
    :param token: Optional API token for authentication.
    :type token: str

    :raises HTTPException: 401 Unauthorized if token is invalid, 500 Internal Server Error if query execution fails.
    :return: A dictionary containing a success message.
    :rtype: Dict[str, str]
    """
    try:
        if token != API_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized.")
        
        _ = delete_chat_history(user_id=user_id)
        return {"message": f"🗑️ Chat history cleared for user={user_id}"}
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


# -------------------- Server Entrypoint --------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
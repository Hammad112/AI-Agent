"""
app/server.py
-------------
FastAPI HTTP server for the AI agent.
Endpoints:
  POST /agent/chat       - Send a message and get a response
  GET  /health           - Health check
  POST /business/load_pdf - Load a new business PDF
"""

import os
import sys
import uuid
from pathlib import Path

# Add parent to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from core.database import init_db, generate_synthetic_data, set_business_meta, get_business_meta
from processing.pdf_processor import process_pdf, load_chunks, save_chunks_to_db
from processing.knowledge_enricher import detect_business_type, enrich_knowledge
from agent.agent import run_agent_turn

app = FastAPI(
    title="Generic AI Customer Service Agent",
    version="1.0.0",
    description="A generic AI agent that handles customer inquiries, orders, and appointments based on a business PDF description.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session cache
_sessions: dict[str, dict] = {}
_all_chunks: list[dict] = []
_business_name: str = ""
_business_type: str = ""


class ChatRequest(BaseModel):
    message: str
    user_id: int = 1
    session_id: str = ""


class ChatResponse(BaseModel):
    response: str
    session_id: str
    detected_intents: list[str] = []
    order_id: int | None = None
    appointment_id: int | None = None
    ics_file_url: str | None = None
    complaint_id: int | None = None


class LoadPDFRequest(BaseModel):
    pdf_path: str


class LoadPDFResponse(BaseModel):
    success: bool
    business_name: str
    business_type: str
    chunks_count: int
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize database on server start."""
    global _all_chunks, _business_name, _business_type

    init_db()

    # Try to load existing data
    _business_name = get_business_meta("business_name") or ""
    _business_type = get_business_meta("business_type") or ""
    if _business_name:
        _all_chunks = load_chunks(_business_name)
        print(f"Loaded business: {_business_name} ({_business_type}), {len(_all_chunks)} chunks")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "business_loaded": bool(_business_name),
        "business_name": _business_name,
        "business_type": _business_type,
        "chunks_count": len(_all_chunks),
    }


@app.post("/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the AI agent and receive a response."""
    global _all_chunks, _business_name, _business_type

    if not _business_name:
        raise HTTPException(
            status_code=400,
            detail="No business PDF loaded. Use POST /business/load_pdf first.",
        )

    if request.session_id and request.session_id in _sessions:
        session_id = request.session_id
    else:
        session_id = request.session_id or str(uuid.uuid4())
        _sessions[session_id] = {}

    try:
        agent_out = run_agent_turn(
            user_id=request.user_id,
            conversation_id=session_id,
            query=request.message,
            all_chunks=_all_chunks,
            business_name=_business_name,
            business_type=_business_type,
        )
        return ChatResponse(
            response=agent_out["response"],
            session_id=session_id,
            detected_intents=agent_out.get("detected_intents", []),
            order_id=agent_out.get("order_id"),
            appointment_id=agent_out.get("appointment_id"),
            ics_file_url=agent_out.get("ics_file_url"),
            complaint_id=agent_out.get("complaint_id"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/business/load_pdf", response_model=LoadPDFResponse)
async def load_pdf(request: LoadPDFRequest):
    """Load and process a new business PDF/document."""
    global _all_chunks, _business_name, _business_type

    file_path = Path(request.pdf_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.pdf_path}")

    try:
        # Process the file
        _all_chunks = process_pdf(str(file_path))

        # Detect business type
        sample_text = "\n".join(c.get("text", "")[:200] for c in _all_chunks[:5])
        business_info = detect_business_type(sample_text)
        _business_name = business_info.get("business_name", "Business")
        _business_type = business_info.get("business_type", "general")

        set_business_meta("business_name", _business_name)
        set_business_meta("business_type", _business_type)
        
        # Save chunks with business name
        save_chunks_to_db(_all_chunks, _business_name)

        # Enrich knowledge
        enrich_knowledge(sample_text, _business_type, _business_name)

        # Generate synthetic data
        generate_synthetic_data(_business_type, _business_name)

        return LoadPDFResponse(
            success=True,
            business_name=_business_name,
            business_type=_business_type,
            chunks_count=len(_all_chunks),
            message=f"Successfully loaded '{_business_name}' with {len(_all_chunks)} knowledge chunks.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

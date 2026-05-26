"""
API routes for SARA system.
Defines endpoints for asking questions about academic regulations.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.services.embedding import get_embedding_service
from app.services.database import get_database_service
from app.services.llm_gemini import get_gemini_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["SARA"])


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    temperature: Optional[float] = Field(0.3, ge=0.0, le=1.0)


class FragmentInfo(BaseModel):
    id: int
    contenido: str
    metadata: Optional[Dict[str, Any]] = None
    similarity: float


class QuestionResponse(BaseModel):
    answer: str
    fragments_used: int
    fragments: List[FragmentInfo]
    model: str


class HealthResponse(BaseModel):
    status: str
    database: bool
    message: str = ""


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    try:
        logger.info(f"Generating embedding for question: {request.question[:50]}...")
        embedding_service = get_embedding_service()
        question_embedding = embedding_service.embed_text(request.question)
        
        logger.info("Searching for similar fragments...")
        db_service = get_database_service()
        fragments = db_service.search_similar_fragments(question_embedding)
        
        if not fragments:
            logger.warning(f"No fragments found")
        else:
            logger.info(f"Found {len(fragments)} fragments")
        
        logger.info("Generating response...")
        gemini_service = get_gemini_service()
        response_data = gemini_service.generate_rag_response(
            question=request.question,
            fragments=fragments,
            temperature=request.temperature
        )
        
        formatted_fragments = [
            FragmentInfo(
                id=f["id"],
                contenido=f["contenido"],
                metadata=f.get("metadata"),
                similarity=f["similarity"]
            )
            for f in fragments
        ]
        
        logger.info("Question processed successfully")
        return QuestionResponse(
            answer=response_data["answer"],
            fragments_used=response_data["fragments_used"],
            fragments=formatted_fragments,
            model="gemini-1.5-flash"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing your question.")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    try:
        db_service = get_database_service()
        db_healthy = db_service.health_check()
        return HealthResponse(
            status="healthy" if db_healthy else "unhealthy",
            database=db_healthy,
            message="All systems operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(status="unhealthy", database=False, message=f"Health check failed: {str(e)}")

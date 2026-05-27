"""
API routes for SARA system.
Defines endpoints for auth, asking questions about academic regulations and TXT uploads.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
from app.services.embedding import get_embedding_service
from app.services.database import get_database_service
from app.services.llm_gemini import get_gemini_service
from app.services.text_normalization import normalize_text
from app.services.s3_storage import get_s3_storage_service
from app.services.auth import authenticate_user, create_access_token, get_current_user, require_admin
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["SARA"])


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    temperature: Optional[float] = Field(0.3, ge=0.0, le=1.0)
    sesion_id: Optional[int] = None


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


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    name: str
    email: str
    role: str


class NormalizeTextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class NormalizeTextResponse(BaseModel):
    original_length: int
    normalized_length: int
    normalized_text: str


class UploadTxtResponse(BaseModel):
    file_name: str
    s3_key: str
    bucket_name: str
    original_length: int
    normalized_length: int


@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    try:
        user = authenticate_user(request.email, request.password)
        token = create_access_token({
            "sub": user["sub"],
            "user_id": user["user_id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
        })
        return LoginResponse(
            access_token=token,
            user_id=user["user_id"],
            name=user["name"],
            email=user["email"],
            role=user["role"] or "user",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed")


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> QuestionResponse:
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
        start_ts = time.time()
        response_data = gemini_service.generate_rag_response(
            question=request.question,
            fragments=fragments,
            temperature=request.temperature
        )
        tiempo_ms = int((time.time() - start_ts) * 1000)

        if request.sesion_id:
            session_owner = db_service.get_session_owner(request.sesion_id)
            if session_owner is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
            if int(session_owner) != int(current_user["user_id"]):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session does not belong to current user")
            sesion_id = request.sesion_id
        else:
            sesion_id = db_service.create_session(int(current_user["user_id"]))

        vector_ids = [f["id"] for f in fragments]
        db_service.save_chat_transaction(
            sesion_id=sesion_id,
            pregunta=request.question,
            respuesta=response_data["answer"],
            vector_ids=vector_ids,
            tiempo_ms=tiempo_ms,
        )
        
        formatted_fragments = [
            FragmentInfo(
                id=f["id"],
                contenido=f.get("contenido_texto") or f.get("contenido"),
                metadata={"titulo": f.get("titulo")},
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


@router.post("/normalize-text", response_model=NormalizeTextResponse)
async def normalize_document_text(request: NormalizeTextRequest) -> NormalizeTextResponse:
    try:
        normalized_text = normalize_text(request.text)
        return NormalizeTextResponse(
            original_length=len(request.text),
            normalized_length=len(normalized_text),
            normalized_text=normalized_text,
        )
    except ValueError as e:
        logger.error(f"Normalization error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {str(e)}")


@router.post("/upload-txt", response_model=UploadTxtResponse)
async def upload_txt_to_s3(
    file: UploadFile = File(...),
    file_name: str = Form(...),
    current_user: Dict[str, Any] = Depends(require_admin),
) -> UploadTxtResponse:
    try:
        if file.content_type not in {"text/plain", "text/txt", "application/octet-stream"}:
            raise ValueError("Only TXT files are allowed")

        if not file.filename or not file.filename.lower().endswith(".txt"):
            raise ValueError("Only TXT files are allowed")

        raw_content = (await file.read()).decode("utf-8")
        normalized_content = normalize_text(raw_content)

        s3_service = get_s3_storage_service()
        s3_key = s3_service.upload_text(file_name=file_name, content=normalized_content)

        return UploadTxtResponse(
            file_name=s3_service.build_object_name(file_name),
            s3_key=s3_key,
            bucket_name=s3_service.bucket_name,
            original_length=len(raw_content),
            normalized_length=len(normalized_content),
        )
    except UnicodeDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="TXT file must be UTF-8 encoded")
    except ValueError as e:
        logger.error(f"TXT upload validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"TXT upload error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error uploading TXT file.")

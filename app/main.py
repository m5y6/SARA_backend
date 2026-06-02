"""
SARA API Main Application
RAG-based API using Google Gemini and pgvector
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from app.core.config import settings
from app.api.routes import router as api_router
from app.services.database import get_database_service
from starlette.concurrency import run_in_threadpool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.api_title,
    description="RAG-based API for academic regulations using Google Gemini and pgvector",
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "name": "SARA",
        "description": "Sistema de Asistencia de Reglamentos Académicos",
        "version": settings.api_version,
        "environment": settings.environment,
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("SARA API Starting...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Database: {settings.db_host}:{settings.db_port}/{settings.db_name}")
    logger.info("=" * 50)

    # Seed `roles` table with default roles if it's empty
    try:
        db = get_database_service()
        default_roles = [
            {"nombre": "admin", "descripcion": "Administrador del sistema"},
            {"nombre": "user", "descripcion": "Usuario estándar"},
            {"nombre": "uploader", "descripcion": "Usuario con permisos de carga"},
        ]
        seeded = await run_in_threadpool(db.seed_roles, default_roles)
        if seeded:
            logger.info("Default roles inserted into 'roles' table.")
        else:
            logger.info("'roles' table already populated.")
    except Exception as e:
        logger.error(f"Role seeding failed: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("SARA API Shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )

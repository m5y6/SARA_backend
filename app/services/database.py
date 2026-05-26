"""
Database service for vector similarity search.
Performs semantic search on academic regulations using pgvector.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
from app.core.config import settings


class DatabaseService:
    """Service for PostgreSQL with pgvector operations."""

    def __init__(self):
        self.db_config = {
            "host": settings.db_host,
            "port": settings.db_port,
            "user": settings.db_user,
            "password": settings.db_pass,
            "database": settings.db_name,
        }

    def _get_connection(self):
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            raise Exception(f"Database connection failed: {str(e)}")

    def search_similar_fragments(
        self,
        query_embedding: List[float],
        top_k: int = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        top_k = top_k or settings.top_k_fragments
        threshold = threshold or settings.similarity_threshold
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT 
                    id,
                    contenido,
                    metadata,
                    (1 - (embedding <=> %s::vector)) as similarity
                FROM fragmentos_vectores
                WHERE (1 - (embedding <=> %s::vector)) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            cursor.execute(query, (embedding_str, embedding_str, threshold, embedding_str, top_k))
            results = cursor.fetchall()
            
            formatted_results = [
                {
                    "id": row["id"],
                    "contenido": row["contenido"],
                    "metadata": row["metadata"],
                    "similarity": float(row["similarity"]),
                }
                for row in results
            ]
            return formatted_results
            
        except psycopg2.Error as e:
            raise Exception(f"Database query failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    def health_check(self) -> bool:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return True
        except psycopg2.Error as e:
            raise Exception(f"Database health check failed: {str(e)}")
        finally:
            if conn:
                conn.close()


_db_service: DatabaseService = None

def get_database_service() -> DatabaseService:
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service

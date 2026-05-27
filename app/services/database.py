"""
Database service for vector similarity search.
Performs semantic search on academic regulations using pgvector.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
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
            conn.autocommit = False
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
                    v.id as id,
                    v.contenido_texto as contenido_texto,
                    d.titulo as titulo,
                    (1 - (v.embedding <=> %s::vector)) as similarity
                FROM fragmentos_vectores v
                JOIN documentos_oficiales d ON v.documento_id = d.id
                WHERE (1 - (v.embedding <=> %s::vector)) >= %s
                ORDER BY v.embedding <=> %s::vector
                LIMIT %s
            """

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            cursor.execute(query, (embedding_str, embedding_str, threshold, embedding_str, top_k))
            results = cursor.fetchall()

            formatted_results = [
                {
                    "id": row["id"],
                    "contenido_texto": row["contenido_texto"],
                    "titulo": row.get("titulo"),
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

    def create_session(self, usuario_id: Optional[int] = None) -> int:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO sesiones_chat (usuario_id) VALUES (%s) RETURNING id", (usuario_id,))
            session_id = cursor.fetchone()[0]
            conn.commit()
            return session_id
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to create session: {str(e)}")
        finally:
            if conn:
                conn.close()

    def save_interaction(self, sesion_id: int, pregunta: str, respuesta: str, tiempo_ms: Optional[int] = None) -> int:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO interacciones (sesion_id, pregunta_usuario, respuesta_ia, tiempo_respuesta_ms) VALUES (%s,%s,%s,%s) RETURNING id",
                (sesion_id, pregunta, respuesta, tiempo_ms)
            )
            interaction_id = cursor.fetchone()[0]
            conn.commit()
            return interaction_id
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to save interaction: {str(e)}")
        finally:
            if conn:
                conn.close()

    def save_references(self, interaccion_id: int, vector_ids: List[int]) -> None:
        if not vector_ids:
            return
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            args_str = ",".join(["(%s,%s)" for _ in vector_ids])
            params = []
            for vid in vector_ids:
                params.extend([interaccion_id, vid])
            cursor.execute(
                f"INSERT INTO referencias_contexto (interaccion_id, vector_id) VALUES {args_str}",
                tuple(params)
            )
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to save references: {str(e)}")
        finally:
            if conn:
                conn.close()

    def save_chat_transaction(
        self,
        sesion_id: int,
        pregunta: str,
        respuesta: str,
        vector_ids: List[int],
        tiempo_ms: Optional[int] = None,
    ) -> int:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO interacciones (sesion_id, pregunta_usuario, respuesta_ia, tiempo_respuesta_ms) VALUES (%s,%s,%s,%s) RETURNING id",
                (sesion_id, pregunta, respuesta, tiempo_ms),
            )
            interaction_id = cursor.fetchone()[0]

            if vector_ids:
                values_sql = ",".join(["(%s,%s)" for _ in vector_ids])
                params = []
                for vector_id in vector_ids:
                    params.extend([interaction_id, vector_id])
                cursor.execute(
                    f"INSERT INTO referencias_contexto (interaccion_id, vector_id) VALUES {values_sql}",
                    tuple(params),
                )

            conn.commit()
            return interaction_id
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to save chat transaction: {str(e)}")
        finally:
            if conn:
                conn.close()


_db_service: DatabaseService = None

def get_database_service() -> DatabaseService:
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service

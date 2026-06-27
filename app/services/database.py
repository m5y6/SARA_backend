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
        similarity_threshold = threshold or settings.similarity_threshold
        distance_threshold = 1 - similarity_threshold

        import logging
        logging.basicConfig(level=logging.INFO)
        logging.info(f"DATABASE SEARCH: similarity_threshold={similarity_threshold}, distance_threshold={distance_threshold}")

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT
                    v.documento_id as id,
                    v.contenido_texto as contenido_texto,
                    COALESCE(v.metadata->>'titulo', v.metadata->>'source') as titulo,
                    (1 - (v.embedding <=> %(embedding)s::vector)) as similarity
                FROM fragmentos_vectores v
                WHERE v.embedding <=> %(embedding)s::vector <= %(distance_threshold)s
                ORDER BY v.embedding <=> %(embedding)s::vector
                LIMIT %(top_k)s
            """

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            params = {
                "embedding": embedding_str,
                "distance_threshold": distance_threshold,
                "top_k": top_k
            }
            
            cursor.execute(query, params)
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

    def create_document_record(
        self,
        titulo: str,
        ruta_s3: str,
        subido_por: Optional[int] = None,
        estado: Optional[str] = "uploaded",
    ) -> int:
        if not titulo or not titulo.strip():
            raise ValueError("Document title cannot be empty")
        if not ruta_s3 or not ruta_s3.strip():
            raise ValueError("S3 path cannot be empty")

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO documentos_oficiales (titulo, ruta_s3, subido_por, estado)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (titulo.strip(), ruta_s3.strip(), subido_por, estado),
            )
            document_id = cursor.fetchone()[0]
            conn.commit()
            return document_id
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to create document record: {str(e)}")
        finally:
            if conn:
                conn.close()

    def update_document_status(self, ruta_s3: str, estado: str) -> None:
        if not ruta_s3 or not ruta_s3.strip():
            raise ValueError("S3 path cannot be empty")
        if not estado or not estado.strip():
            raise ValueError("Status cannot be empty")

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE documentos_oficiales
                SET estado = %s, fecha_actualizacion = CURRENT_TIMESTAMP
                WHERE ruta_s3 = %s
                """,
                (estado.strip(), ruta_s3.strip()),
            )
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to update document status: {str(e)}")
        finally:
            if conn:
                conn.close()

    def _get_fragment_document_column(self, cursor) -> Optional[str]:
        candidate_columns = [
            "documento_id",
            "document_id",
            "id_documento",
            "doc_id",
            "documento_oficial_id",
        ]
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'fragmentos_vectores'
            """
        )
        existing_columns = {row[0] for row in cursor.fetchall()}
        for column_name in candidate_columns:
            if column_name in existing_columns:
                return column_name
        return None

    def delete_document_by_s3_key(self, s3_key: str) -> int:
        if not s3_key or not s3_key.strip():
            raise ValueError("S3 key cannot be empty")

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            fragment_column = self._get_fragment_document_column(cursor)
            if fragment_column:
                cursor.execute(
                    f"""
                    DELETE FROM fragmentos_vectores
                    WHERE {fragment_column} IN (
                        SELECT id FROM documentos_oficiales WHERE ruta_s3 = %s
                    )
                    """,
                    (s3_key.strip(),),
                )
            cursor.execute("DELETE FROM documentos_oficiales WHERE ruta_s3 = %s", (s3_key.strip(),))
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to delete document: {str(e)}")
        finally:
            if conn:
                conn.close()

    def seed_roles(self, default_roles: Optional[List[Dict[str, str]]] = None) -> bool:
        """Ensure the `roles` table has the default roles.

        If the table is empty, insert the provided `default_roles` (list of dicts with
        `nombre` and optional `descripcion`). Returns True if seeding occurred.
        """
        roles_to_insert = default_roles or [
            {"nombre": "admin", "descripcion": "Administrador del sistema"},
            {"nombre": "user", "descripcion": "Usuario estándar"},
            {"nombre": "uploader", "descripcion": "Usuario con permisos de carga"},
        ]

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Prefer the `roles` table (plural) used on RDS; create if missing.
            table_name = 'roles'
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s)",
                (table_name,)
            )
            exists = cursor.fetchone()[0]
            if not exists:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        nombre VARCHAR(255) UNIQUE NOT NULL,
                        descripcion TEXT
                    )
                    """
                )
                conn.commit()

            # Ensure expected columns exist; add them if missing to support older schemas.
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s)",
                (table_name, 'nombre'),
            )
            nombre_exists = cursor.fetchone()[0]
            if not nombre_exists:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN nombre VARCHAR(255)")

            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s)",
                (table_name, 'descripcion'),
            )
            descripcion_exists = cursor.fetchone()[0]
            if not descripcion_exists:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN descripcion TEXT")
            conn.commit()

            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            if count and int(count) > 0:
                return False

            for r in roles_to_insert:
                nombre = r.get("nombre")
                descripcion = r.get("descripcion")
                cursor.execute(f"SELECT 1 FROM {table_name} WHERE nombre = %s", (nombre,))
                if cursor.fetchone():
                    continue
                cursor.execute(
                    f"INSERT INTO {table_name} (nombre, descripcion) VALUES (%s, %s)",
                    (nombre, descripcion),
                )
            conn.commit()
            return True
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to seed roles: {str(e)}")
        finally:
            if conn:
                conn.close()
    def list_roles(self) -> List[Dict[str, Any]]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT id, nombre, permisos FROM roles ORDER BY id")
            return cursor.fetchall()
        except psycopg2.Error as e:
            raise Exception(f"Failed to list roles: {str(e)}")
        finally:
            if conn:
                conn.close()

    def list_users(self) -> List[Dict[str, Any]]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT u.id, u.nombre, u.email, u.rol_id, r.nombre AS role_name FROM usuarios u LEFT JOIN roles r ON u.rol_id = r.id ORDER BY u.id"
            )
            return cursor.fetchall()
        except psycopg2.Error as e:
            raise Exception(f"Failed to list users: {str(e)}")
        finally:
            if conn:
                conn.close()

    def get_role_by_name(self, role_name: str) -> Optional[Dict[str, Any]]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT id, nombre, permisos FROM roles WHERE nombre = %s", (role_name,))
            return cursor.fetchone()
        except psycopg2.Error as e:
            raise Exception(f"Failed to get role by name: {str(e)}")
        finally:
            if conn:
                conn.close()

    def update_user_role(self, user_id: int, role_id: Optional[int], performed_by: Optional[int] = None) -> None:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE usuarios SET rol_id = %s WHERE id = %s", (role_id, user_id))
            # audit
            cursor.execute(
                "INSERT INTO role_audit_log (action, performed_by, target_user, role_id, details) VALUES (%s,%s,%s,%s,%s)",
                ("assign_role", performed_by, user_id, role_id, None),
            )
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to update user role: {str(e)}")
        finally:
            if conn:
                conn.close()

    def update_role_permisos(self, role_id: int, permisos: Any, performed_by: Optional[int] = None) -> None:
        import json

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            permisos_json = json.dumps(permisos) if permisos is not None else None
            cursor.execute("UPDATE roles SET permisos = %s WHERE id = %s", (permisos_json, role_id))
            cursor.execute(
                "INSERT INTO role_audit_log (action, performed_by, target_user, role_id, details) VALUES (%s,%s,%s,%s,%s)",
                ("update_role_permisos", performed_by, None, role_id, permisos_json),
            )
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to update role permisos: {str(e)}")
        finally:
            if conn:
                conn.close()

    def list_role_audit(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM role_audit_log ORDER BY created_at DESC LIMIT %s", (limit,))
            return cursor.fetchall()
        except psycopg2.Error as e:
            raise Exception(f"Failed to list role audit log: {str(e)}")
        finally:
            if conn:
                conn.close()

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT u.id, u.nombre, u.email, u.rol_id, r.nombre AS role_name FROM usuarios u LEFT JOIN roles r ON u.rol_id = r.id WHERE u.id = %s",
                (user_id,)
            )
            return cursor.fetchone()
        except psycopg2.Error as e:
            raise Exception(f"Failed to get user: {str(e)}")
        finally:
            if conn:
                conn.close()

    def update_user(self, user_id: int, nombre: Optional[str] = None, email: Optional[str] = None, role_id: Optional[int] = None, performed_by: Optional[int] = None) -> None:
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            updates = []
            params = []
            if nombre is not None:
                updates.append("nombre = %s")
                params.append(nombre)
            if email is not None:
                updates.append("email = %s")
                params.append(email)

            if updates:
                params.extend([user_id])
                sql = f"UPDATE usuarios SET {', '.join(updates)} WHERE id = %s"
                cursor.execute(sql, tuple(params))

            # Handle role change on the same transaction to avoid locking the same row twice.
            if role_id is not None:
                cursor.execute("UPDATE usuarios SET rol_id = %s WHERE id = %s", (role_id, user_id))
                cursor.execute(
                    "INSERT INTO role_audit_log (action, performed_by, target_user, role_id, details) VALUES (%s,%s,%s,%s,%s)",
                    ("assign_role", performed_by, user_id, role_id, None),
                )

            # audit update of user fields
            details = {
                "updated_fields": {
                    k.split(' = ')[0]: v for k, v in zip(updates, params[:len(updates)])
                }
            } if updates else None

            if details:
                import json
                cursor.execute(
                    "INSERT INTO role_audit_log (action, performed_by, target_user, role_id, details) VALUES (%s,%s,%s,%s,%s)",
                    ("update_user", performed_by, user_id, None, json.dumps(details)),
                )

            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to update user: {str(e)}")
        finally:
            if conn:
                conn.close()


_db_service: DatabaseService = None

def get_database_service() -> DatabaseService:
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service

"""
Initialize database schema for SARA.
Runs the provided DDL to create required tables and extensions.
Usage: python -m app.db.init_db
"""
import sys
import logging
from app.core.config import settings
import psycopg2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DDL = [
    "CREATE EXTENSION IF NOT EXISTS vector;",
    """
    CREATE TABLE IF NOT EXISTS roles (
        id SERIAL PRIMARY KEY,
        nombre VARCHAR(20) NOT NULL,
        permisos JSONB
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS usuarios (
        id SERIAL PRIMARY KEY,
        rol_id INT REFERENCES roles(id) ON DELETE SET NULL,
        nombre VARCHAR(20) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS documentos_oficiales (
        id SERIAL PRIMARY KEY,
        titulo VARCHAR(255) NOT NULL,
        ruta_s3 VARCHAR(500),
        subido_por INT REFERENCES usuarios(id) ON DELETE SET NULL,
        fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        estado VARCHAR(50)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS fragmentos_vectores (
        id SERIAL PRIMARY KEY,
        documento_id INT REFERENCES documentos_oficiales(id) ON DELETE CASCADE,
        contenido_texto TEXT NOT NULL,
        embedding vector(384),
        numero_pagina INT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sesiones_chat (
        id SERIAL PRIMARY KEY,
        usuario_id INT REFERENCES usuarios(id) ON DELETE CASCADE,
        fecha_inicio TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS interacciones (
        id SERIAL PRIMARY KEY,
        sesion_id INT REFERENCES sesiones_chat(id) ON DELETE CASCADE,
        pregunta_usuario TEXT NOT NULL,
        respuesta_ia TEXT NOT NULL,
        tiempo_respuesta_ms INT,
        fecha_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS referencias_contexto (
        id SERIAL PRIMARY KEY,
        interaccion_id INT REFERENCES interacciones(id) ON DELETE CASCADE,
        vector_id INT REFERENCES fragmentos_vectores(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS role_audit_log (
        id SERIAL PRIMARY KEY,
        action VARCHAR(100) NOT NULL,
        performed_by INT REFERENCES usuarios(id) ON DELETE SET NULL,
        target_user INT REFERENCES usuarios(id) ON DELETE SET NULL,
        role_id INT REFERENCES roles(id) ON DELETE SET NULL,
        details JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
]


def run():
    logger.info("Connecting to DB: %s", settings.database_url)
    conn = None
    try:
        conn = psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_pass,
            dbname=settings.db_name,
        )
        conn.autocommit = True
        cur = conn.cursor()
        for stmt in DDL:
            logger.info("Executing statement...")
            cur.execute(stmt)
        cur.close()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize database: %s", e)
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    run()

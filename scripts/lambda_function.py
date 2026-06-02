import json
import os
import re
import shutil
import sys
import unicodedata
import urllib.parse
import zipfile

import boto3
import pg8000.native

sys.path.append('/opt/python')
from fastembed import TextEmbedding


BUCKET_NAME = "sara-repository-duoc"
MODEL_ZIP_KEY = "modelo_sara (1).zip"
TMP_BASE = "/tmp/fastembed_cache"

embedding_model = None


def normalize_text(text: str) -> str:
    if text is None:
        raise ValueError("Text cannot be empty")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(\w)-\n(\w)", r"\1\2", normalized)
    normalized = re.sub(r"[\t\f\v]+", " ", normalized)
    normalized = re.sub(r"[ ]{2,}", " ", normalized)

    lines = []
    previous_was_empty = False
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            if not previous_was_empty:
                lines.append("")
            previous_was_empty = True
            continue

        line = re.sub(r"\s{2,}", " ", line)
        lines.append(line)
        previous_was_empty = False

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def prepare_document_text(text: str) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(r"(?m)^\s*Page \d+\s*$", "", normalized)
    normalized = re.sub(r"(?m)^\s*\d+\s*$", "", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"\n{2,}", "\n\n", normalized)
    return normalized.strip()


def ensure_model_is_ready():
    global embedding_model
    if embedding_model is not None:
        return

    flat_path = os.path.join(TMP_BASE, "bge-small-en-v1.5")

    if not os.path.exists(flat_path):
        print("LOG: Instalando cerebro desde S3...")
        os.makedirs(TMP_BASE, exist_ok=True)
        s3 = boto3.client("s3")
        local_zip = "/tmp/temp_model.zip"

        s3.download_file(BUCKET_NAME, MODEL_ZIP_KEY, local_zip)
        extract_path = "/tmp/extrayendo"
        with zipfile.ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(local_zip)

        onnx_folder = None
        for root, dirs, files in os.walk(extract_path):
            if any(f.endswith(".onnx") for f in files):
                onnx_folder = root
                break

        if onnx_folder:
            print(f"LOG: Modelo encontrado en {onnx_folder}. Aplanando a {flat_path}...")
            os.makedirs(flat_path, exist_ok=True)
            for item in os.listdir(onnx_folder):
                shutil.move(os.path.join(onnx_folder, item), os.path.join(flat_path, item))
            shutil.rmtree(extract_path)
        else:
            raise Exception("No se encontró ningún archivo .onnx en el ZIP de S3.")

    print("LOG: Inicializando motor de IA local...")
    embedding_model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=TMP_BASE,
        local_files_only=True,
    )


def get_embedding_local(text):
    try:
        ensure_model_is_ready()
        embeddings_generator = embedding_model.embed([text])
        vector = list(next(embeddings_generator))
        return [float(v) for v in vector]
    except Exception as e:
        print(f"Error en vectorización local: {e}")
        return None


def ensure_document_record(conn, key: str):
    doc_row = conn.run(
        "SELECT id FROM documentos_oficiales WHERE ruta_s3 = :ruta_s3 ORDER BY id ASC LIMIT 1",
        ruta_s3=key,
    )

    if doc_row:
        return doc_row[0][0]

    titulo = os.path.basename(key)
    print(f"LOG: No existe documento_oficiales para ruta_s3={key}. Creando registro...")
    inserted = conn.run(
        """
        INSERT INTO documentos_oficiales (titulo, ruta_s3, estado)
        VALUES (:titulo, :ruta_s3, :estado)
        RETURNING id
        """,
        titulo=titulo,
        ruta_s3=key,
        estado="processing",
    )
    return inserted[0][0]


def update_document_status(conn, document_id: int, estado: str):
    conn.run(
        """
        UPDATE documentos_oficiales
        SET estado = :estado,
            fecha_actualizacion = CURRENT_TIMESTAMP
        WHERE id = :documento_id
        """,
        estado=estado,
        documento_id=document_id,
    )


def lambda_handler(event, context):
    print("LOG: Iniciando SARA Procesador")

    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    raw_key = event["Records"][0]["s3"]["object"]["key"]
    key = urllib.parse.unquote_plus(raw_key)

    if key.endswith(".zip"):
        return {"status": "skipped"}

    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    file_content = response["Body"].read().decode("utf-8-sig")

    texto_limpio = prepare_document_text(file_content)
    chunks = [texto_limpio[i:i + 1000] for i in range(0, len(texto_limpio), 800)]
    print(f"Procesando {len(chunks)} fragmentos de {key}")

    conn = None
    documento_id = None
    try:
        conn = pg8000.native.Connection(
            user="postgres",
            host=os.environ["bd"],
            database="sara_db",
            password=os.environ["db_password"],
        )

        documento_id = ensure_document_record(conn, key)

        for chunk in chunks:
            vector = get_embedding_local(chunk)
            if vector:
                print("DEBUG: Insertando vector (384 dimensiones)")
                sql = """
                INSERT INTO fragmentos_vectores
                (documento_id, contenido_texto, embedding, metadata)
                VALUES (:documento_id, :contenido_texto, :embedding, :metadata)
                """

                conn.run(
                    sql,
                    documento_id=documento_id,
                    contenido_texto=chunk,
                    embedding=str(vector),
                    metadata=json.dumps(
                        {
                            "source": key,
                            "titulo": os.path.basename(key),
                        }
                    ),
                )

        if documento_id is not None:
            update_document_status(conn, documento_id, "vectorized")

        print("LOG: ¡SARA COMPLETADA! Todos los vectores guardados.")
        return {"status": "success", "file": key}

    except Exception as e:
        print(f"ERROR: {e}")
        if conn and documento_id is not None:
            try:
                update_document_status(conn, documento_id, "failed")
            except Exception as status_error:
                print(f"ERROR actualizando estado del documento: {status_error}")
        raise

    finally:
        if conn:
            conn.close()
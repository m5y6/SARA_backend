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

# --- Configuration ---
# The S3 bucket where the model zip file is stored.
BUCKET_NAME = "sara-repository-duoc"
# The name of the multilingual model being used.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# The S3 key (filename) of the zip file containing the model.
# IMPORTANT: The user must create this zip file and upload it to the S3 bucket.
MODEL_S3_KEY = "paraphrase-multilingual-mpnet-base-v2.zip"
# A temporary directory within the Lambda environment for caching model files.
TMP_BASE = "/tmp/fastembed_cache"
# The expected local path where the model files will be placed.
MODEL_LOCAL_PATH = os.path.join(TMP_BASE, MODEL_NAME)

# Global variable to hold the initialized embedding model.
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
    """
    Ensures the embedding model is initialized from a local S3 download.
    This avoids the need for internet access in the Lambda environment.
    """
    global embedding_model
    if embedding_model is not None:
        return

    # Check if model files are already available locally in the /tmp directory.
    # We check for a common model file like 'model.onnx'.
    if not os.path.exists(os.path.join(MODEL_LOCAL_PATH, "model.onnx")):
        print(f"LOG: Model not found locally. Attempting to download from S3 (s3://{BUCKET_NAME}/{MODEL_S3_KEY}).")
        os.makedirs(MODEL_LOCAL_PATH, exist_ok=True)
        s3 = boto3.client("s3")
        local_zip_path = f"/tmp/{MODEL_S3_KEY}"

        try:
            s3.download_file(BUCKET_NAME, MODEL_S3_KEY, local_zip_path)
            print("LOG: Model zip file downloaded successfully from S3.")
        except Exception as e:
            print(f"FATAL: Failed to download model from S3 bucket '{BUCKET_NAME}' with key '{MODEL_S3_KEY}'. Please ensure the file exists and the Lambda has S3 permissions. Error: {e}")
            raise

        # Unzip the model files.
        extract_path = "/tmp/extracting_model"
        print(f"LOG: Unzipping model from {local_zip_path} to {extract_path}...")
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(local_zip_path)
        print("LOG: Unzip complete.")

        # The actual model files might be nested in a subdirectory inside the zip.
        # We walk the extracted path to find the directory containing 'model.onnx'.
        onnx_folder = extract_path
        for root, dirs, files in os.walk(extract_path):
            if "model.onnx" in files:
                onnx_folder = root
                break
        
        print(f"LOG: Found model files in '{onnx_folder}'. Moving them to final destination: '{MODEL_LOCAL_PATH}'")
        # Move all files from the found folder to the target cache directory.
        for item in os.listdir(onnx_folder):
            shutil.move(os.path.join(onnx_folder, item), os.path.join(MODEL_LOCAL_PATH, item))
        shutil.rmtree(extract_path)

    print("LOG: Initializing embedding model from local files...")
    embedding_model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=TMP_BASE,
        local_files_only=True,  # This is CRITICAL to prevent internet access attempts.
    )
    print("LOG: Model initialized successfully.")


def get_embedding_local(text: str) -> list[float] | None:
    try:
        ensure_model_is_ready()
        embeddings_generator = embedding_model.embed([text])
        vector = list(next(embeddings_generator))
        return [float(v) for v in vector]
    except Exception as e:
        print(f"Error during local vectorization: {e}")
        # Re-raise the exception to make the Lambda fail and retry, which might resolve transient issues.
        raise


def ensure_document_record(conn, key: str):
    doc_row = conn.run(
        "SELECT id FROM documentos_oficiales WHERE ruta_s3 = :ruta_s3 ORDER BY id ASC LIMIT 1",
        ruta_s3=key,
    )

    if doc_row:
        return doc_row[0][0]

    titulo = os.path.basename(key)
    print(f"LOG: No 'documentos_oficiales' record for ruta_s3={key}. Creating one...")
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
    print("LOG: SARA Processor starting...")

    try:
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        raw_key = event["Records"][0]["s3"]["object"]["key"]
        key = urllib.parse.unquote_plus(raw_key)

        if key.endswith(".zip"):
            print(f"LOG: Skipping file '{key}' as it appears to be a zip archive.")
            return {"status": "skipped"}

        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_content = response["Body"].read().decode("utf-8-sig")

        texto_limpio = prepare_document_text(file_content)
        chunks = [texto_limpio[i : i + 1000] for i in range(0, len(texto_limpio), 800)]
        print(f"Processing {len(chunks)} fragments from '{key}'")

        conn = None
        documento_id = None
        
        conn = pg8000.native.Connection(
            user=os.environ.get("db_user", "postgres"),
            host=os.environ["bd"],
            database=os.environ.get("db_name", "sara_db"),
            password=os.environ["db_password"],
        )

        documento_id = ensure_document_record(conn, key)

        for chunk in chunks:
            vector = get_embedding_local(chunk)
            if vector:
                print("DEBUG: Inserting vector (768 dimensions)")
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
                    metadata=json.dumps({"source": key, "titulo": os.path.basename(key)}),
                )

        if documento_id is not None:
            update_document_status(conn, documento_id, "vectorized")

        print("LOG: SARA processing complete! All vectors saved.")
        return {"status": "success", "file": key}

    except Exception as e:
        print(f"FATAL ERROR in lambda_handler: {e}")
        # Attempt to mark the document as failed if we got far enough to have a DB connection and document ID.
        if 'conn' in locals() and conn and 'documento_id' in locals() and documento_id:
            try:
                update_document_status(conn, documento_id, "failed")
            except Exception as status_error:
                print(f"ERROR: Could not update document status to 'failed'.Nested error: {status_error}")
        raise

    finally:
        if 'conn' in locals() and conn:
            conn.close()
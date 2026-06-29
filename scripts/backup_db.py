"""
Script de Respaldo Automatizado para PostgreSQL y S3

Este script realiza una copia de seguridad (dump) de una base de datos PostgreSQL,
la comprime y la sube a un bucket de AWS S3.

Requisitos:
- tener las herramientas de cliente de PostgreSQL instaladas (específicamente `pg_dump`).
- `boto3` y `python-dotenv` instalados en el entorno (`pip install boto3 python-dotenv`).
- Un archivo `.env` con las credenciales o las variables de entorno definidas.
"""
import os
import subprocess
import sys
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv

def main():
    """
    Función principal que orquesta el proceso de respaldo y subida.
    """
    load_dotenv()

    db_name = os.environ.get("DB_NAME")
    db_user = os.environ.get("DB_USER")
    db_pass = os.environ.get("DB_PASS")
    db_host = os.environ.get("DB_HOST")
    db_port = os.environ.get("DB_PORT", "5432")

    s3_bucket_name = os.environ.get("S3_BUCKET_NAME")
    s3_prefix = os.environ.get("S3_BACKUP_PREFIX", "backups/")

    if not all([db_name, db_user, db_pass, db_host, s3_bucket_name]):
        print("Error: Faltan variables de entorno críticas (DB_NAME, DB_USER, etc.).")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_filename = f"backup-{timestamp}.dump"

    backup_filepath = f"/tmp/{backup_filename}" 

    print(f"Iniciando respaldo de la base de datos '{db_name}'...")
    print(f"Archivo de respaldo temporal: {backup_filepath}")

    command = [
        'pg_dump',
        '--dbname', f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}",
        '-Fc',
        '-v',
        '--file', backup_filepath
    ]

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        
        print("\n--- Salida de pg_dump (stdout) ---")
        print(process.stdout)
        
        print("\n--- Salida de pg_dump (stderr) ---")
        print(process.stderr)
        
        print(f"Respaldo de la base de datos completado exitosamente: {backup_filename}")

    except FileNotFoundError:
        print("Error: El comando 'pg_dump' no se encontró. Asegúrate de que las herramientas de cliente de PostgreSQL estén instaladas y en el PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error durante la ejecución de pg_dump (código de salida {e.returncode}):")
        print(e.stderr) 
        sys.exit(1)
    except Exception as e:
        print(f"Un error inesperado ocurrió durante el respaldo: {e}")
        sys.exit(1)

    s3_key = f"{s3_prefix}{backup_filename}"
    
    print(f"\nSubiendo respaldo a S3: s3://{s3_bucket_name}/{s3_key}")

    try:
        s3_client = boto3.client('s3')
        
        s3_client.upload_file(backup_filepath, s3_bucket_name, s3_key)
        
        print("Subida a S3 completada exitosamente.")

    except (NoCredentialsError, PartialCredentialsError):
        print("Error de credenciales de AWS. Configura tus credenciales (ej. variables de entorno AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).")
        sys.exit(1)
    except Exception as e:
        print(f"Error durante la subida a S3: {e}")
        sys.exit(1)
    
    finally:
        if os.path.exists(backup_filepath):
            print(f"Limpiando archivo de respaldo local: {backup_filepath}")
            os.remove(backup_filepath)

    print("\n¡Proceso de respaldo y subida finalizado con éxito!")

if __name__ == "__main__":
    main()
"""
Archivo de pruebas automatizadas para la API de SARA.

Utiliza pytest y TestClient de FastAPI para validar los endpoints,
el manejo de errores y la seguridad de las entradas.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Importar la app de FastAPI
# Asegúrate de que la app pueda ser importada.
# Si la estructura de tu proyecto es `backend-SARA/app/main.py`,
# necesitarás tener un `__init__.py` en las carpetas para que Python las trate como paquetes.
from app.main import app
from app.api.routes import get_current_user


# --- Dependency Override ---
def get_mock_current_user():
    """Función mock para simular un usuario autenticado."""
    return {"user_id": 1, "name": "Test User", "role": "user", "sub": "test@example.com"}


# Sobrescribir la dependencia `get_current_user` en la app
app.dependency_overrides[get_current_user] = get_mock_current_user


# Creación de un cliente de prueba para la API
client = TestClient(app)

# --- Casos de Prueba Automatizados ---

def test_health_check_returns_200_ok():
    """
    Caso 1: Validación de un endpoint respondiendo 200 OK.
    Verifica que el endpoint de health check esté operativo y responda correctamente.
    """
    # Se simula una respuesta exitosa del servicio de base de datos.
    with patch('app.api.routes.get_database_service') as mock_db_service:
        mock_db_instance = MagicMock()
        mock_db_instance.health_check.return_value = True
        mock_db_service.return_value = mock_db_instance

        # Se realiza la petición al endpoint /api/v1/health
        response = client.get("/api/v1/health")

        # Se comprueba que el código de estado sea 200 (OK)
        assert response.status_code == 200
        # Se comprueba que el contenido de la respuesta sea el esperado
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] is True


def test_chat_endpoint_with_valid_input():
    """
    Caso 2: Validación de entrada de datos correcta.
    Simula una petición válida al endpoint de chat y verifica que la respuesta sea exitosa.
    Aquí se "mockean" (simulan) los servicios externos para aislar la lógica del endpoint.
    """
    # Se definen los datos de la petición de chat
    chat_request_data = {"question": "¿Cuál es el reglamento de prácticas?"}

    # Se utiliza 'patch' para simular los servicios externos (base de datos, embedding, LLM)
    with patch('app.api.routes.get_database_service') as mock_db_service:
        with patch('app.api.routes.get_embedding_service') as mock_embedding_service:
            with patch('app.api.routes.get_gemini_service') as mock_gemini_service:
                # Configuración de los mocks
                mock_db_instance = MagicMock()
                mock_db_instance.get_session_owner.return_value = 1 # El usuario es dueño de la sesión
                mock_db_instance.create_session.return_value = 123  # ID de sesión nueva
                mock_db_instance.get_chat_history.return_value = [] # Sin historial previo
                mock_db_instance.search_similar_fragments.return_value = [{"id": 1, "contenido": "Fragmento de prueba", "similarity": 0.9}]
                mock_db_service.return_value = mock_db_instance
                
                mock_embedding_instance = MagicMock()
                mock_embedding_instance.embed_text.return_value = [0.1] * 768 # Vector de embedding simulado
                mock_embedding_service.return_value = mock_embedding_instance

                mock_gemini_instance = MagicMock()
                mock_gemini_instance.model_name = "gemini-test-model" # <-- FIX: Add model_name
                mock_gemini_instance.generate_rag_response.return_value = {"answer": "La respuesta es 42.", "fragments_used": 1}
                mock_gemini_service.return_value = mock_gemini_instance

                # Se realiza la petición POST al endpoint /api/v1/chat
                response = client.post("/api/v1/chat", json=chat_request_data)

                # Se comprueba que el código de estado sea 200 (OK)
                assert response.status_code == 200
                # Se comprueba que la respuesta contenga los campos esperados
                data = response.json()
                assert "answer" in data
                assert data["answer"] == "La respuesta es 42."
                assert "sesion_id" in data
                assert data["sesion_id"] == 123


def test_chat_endpoint_simulated_db_error():
    """
    Caso 3: Manejo de un error simulado (base de datos caída).
    Verifica que la API maneje correctamente un error interno y devuelva un código 500.
    """
    chat_request_data = {"question": "Una pregunta que fallará"}

    # Se simula que el servicio de base de datos lanza una excepción
    with patch('app.api.routes.get_database_service') as mock_db_service:
        mock_db_instance = MagicMock()
        # Se configura el mock para que lance una excepción cuando se use
        mock_db_instance.create_session.side_effect = Exception("¡La base de datos no responde!")
        mock_db_service.return_value = mock_db_instance

        # Se realiza la petición
        response = client.post("/api/v1/chat", json=chat_request_data)

        # Se comprueba que el código de estado sea 500 (Error Interno del Servidor)
        assert response.status_code == 500
        # Se comprueba que el mensaje de error sea genérico para no exponer detalles internos
        assert response.json() == {"detail": "Error processing your request."}


def test_chat_endpoint_malicious_input():
    """
    Caso 4: Bloqueo de entrada de datos maliciosa.
    Verifica que la validación de Pydantic rechace una petición con datos inválidos.
    En este caso, una pregunta que excede la longitud máxima permitida (500 caracteres).
    """
    # Se genera un string de más de 500 caracteres
    long_question = "a" * 501
    chat_request_data = {"question": long_question}

    # Se realiza la petición con datos inválidos
    response = client.post("/api/v1/chat", json=chat_request_data)

    # Se comprueba que el código de estado sea 422 (Unprocessable Entity)
    # FastAPI usa este código para errores de validación de Pydantic.
    assert response.status_code == 422
    
    # Opcional: verificar el detalle del error
    data = response.json()
    assert "detail" in data
    assert data["detail"][0]["msg"].startswith("String should have at most 500 characters")

def test_chat_endpoint_empty_input():
    """
    Caso extra: Bloqueo de entrada de datos maliciosa (pregunta vacía).
    Verifica que la validación rechace una pregunta vacía.
    """
    chat_request_data = {"question": ""}

    response = client.post("/api/v1/chat", json=chat_request_data)
    
    # El código debe ser 422 por la validación de Pydantic (min_length=1)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"][0]["msg"].startswith("String should have at least 1 character")

import jwt
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

from app.core.config import settings
from app.services.database import get_database_service

# Esquema de seguridad OAuth2 y contexto de contraseñas
# tokenUrl debe apuntar al endpoint de login
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# --- Funciones de Utilidad de Contraseñas ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica una contraseña en texto plano contra un hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Genera un hash para una contraseña en texto plano."""
    return pwd_context.hash(password)


# --- Funciones de Autenticación y Token JWT ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Crea un nuevo token de acceso JWT.
    El token incluye los 'claims' del diccionario de datos y una fecha de expiración.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def authenticate_user(email: str, password: str) -> Dict[str, Any]:
    """
    Autentica un usuario por email y contraseña contra la base de datos PostgreSQL.

    Args:
        email: Email del usuario.
        password: Contraseña del usuario.

    Raises:
        HTTPException: Si la autenticación falla (usuario no encontrado o contraseña incorrecta).
        
    Returns:
        Un diccionario con los datos del usuario para ser incluidos en el token.
    """
    db = get_database_service()  # Obtiene una instancia del servicio de base de datos
    user = db.get_user_for_auth(email)
    
    if not user or not verify_password(password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El correo electrónico o la contraseña son incorrectos.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Prepara los datos del usuario para la carga útil (payload) del token
    user_data_for_token = {
        "sub": user["email"],  # El 'subject' del token es el email
        "user_id": user["id"],
        "name": user["nombre"],
        "email": user["email"],
        "role": user.get("role_name"),
        "permisos": user.get("permisos")
    }
    return user_data_for_token

# Alias para compatibilidad con diferentes versiones del router.
login_user = authenticate_user


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Dependencia de FastAPI para decodificar el token JWT y obtener el usuario actual.
    Eleva HTTPException si el token es inválido, ha expirado o las credenciales no son válidas.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        
        # El payload del token es la fuente de verdad. No se requiere una llamada
        # a la base de datos en cada petición protegida.
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El token ha expirado.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise credentials_exception


# --- Dependencias de Control de Acceso Basado en Roles (RBAC) ---

def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Dependencia que verifica si el usuario actual tiene el rol 'admin'.
    Eleva HTTPException 403 si el usuario no es administrador.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Se requiere acceso de administrador.",
        )
    return current_user


def require_permission(permission_name: str):
    """
    Factoría que crea una dependencia para verificar un permiso específico.
    Los administradores ('admin') tienen acceso a todos los permisos por defecto.
    """
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        # Si el rol es admin, se concede el acceso automáticamente
        if current_user.get("role") == "admin":
            return current_user

        permisos = current_user.get("permisos")
        if not isinstance(permisos, dict) or not permisos.get(permission_name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Se requiere el permiso '{permission_name}'.",
            )
        return current_user
    
    return permission_checker


def require_admin_or_permission(permission_name: str):
    """
    Factoría que crea una dependencia para verificar si un usuario es admin O tiene un permiso específico.
    """
    def admin_or_permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        is_admin = current_user.get("role") == "admin"
        
        permisos = current_user.get("permisos")
        has_perm = isinstance(permisos, dict) and permisos.get(permission_name)

        if not is_admin and not has_perm:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Se requiere acceso de administrador o el permiso '{permission_name}'.",
            )
        return current_user

    return admin_or_permission_checker

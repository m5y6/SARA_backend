"""
Authentication helpers for SARA.
Provides password verification, JWT creation, and current user lookups.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from app.core.config import settings
from app.services.database import get_database_service


pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_minutes: Optional[int] = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes or settings.access_token_expire_minutes)
    payload.update({"exp": expire})
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> Dict[str, Any]:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_access_token(credentials.credentials)
    return payload


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    # Admin can be granted either by role name or by a permission flag in `permisos`.
    role = current_user.get("role")
    permisos = current_user.get("permisos")
    is_admin = role == "admin"
    if not is_admin:
        # check permisos: dict with 'admin': true, or list containing 'admin'
        if isinstance(permisos, dict):
            is_admin = bool(permisos.get("admin"))
        elif isinstance(permisos, (list, tuple)):
            is_admin = "admin" in permisos

    if not is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return current_user


def require_permission(permission: str):
    def _dependency(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        permisos = current_user.get("permisos")
        if not permisos:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{permission}' required")
        if isinstance(permisos, dict):
            if permisos.get(permission):
                return current_user
        elif isinstance(permisos, (list, tuple)):
            if permission in permisos:
                return current_user
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{permission}' required")

    return _dependency


def require_admin_or_permission(permission: str):
    def _dependency(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        # Admin always allowed
        if current_user.get("role") == "admin":
            return current_user
        # Otherwise check permisos
        permisos = current_user.get("permisos")
        if not permisos:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{permission}' required")
        if isinstance(permisos, dict):
            if permisos.get(permission):
                return current_user
        elif isinstance(permisos, (list, tuple)):
            if permission in permisos:
                return current_user
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{permission}' required")

    return _dependency


def authenticate_user(email: str, password: str) -> Dict[str, Any]:
    db_service = get_database_service()
    conn = None
    try:
        conn = db_service._get_connection()
        cursor = conn.cursor(cursor_factory=None)
        cursor.execute(
            """
            SELECT u.id, u.nombre, u.email, u.password_hash, r.nombre AS role_name, r.permisos AS role_permisos
            FROM usuarios u
            LEFT JOIN roles r ON u.rol_id = r.id
            WHERE u.email = %s
            """,
            (email,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        user_id, nombre, user_email, password_hash, role_name, role_permisos = row
        try:
            password_valid = verify_password(password, password_hash)
        except Exception:
            # Stored hash is invalid/corrupted or not from passlib; treat as bad credentials.
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        if not password_valid:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        role_slug = (role_name or "").strip().lower()
        if role_slug in {"admin", "administrador"}:
            role_slug = "admin"
        return {
            "sub": str(user_id),
            "user_id": user_id,
            "name": nombre,
            "email": user_email,
            "role": role_slug,
            "permisos": role_permisos,
        }
    finally:
        if conn:
            conn.close()
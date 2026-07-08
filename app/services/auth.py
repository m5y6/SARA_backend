import sqlite3
import bcrypt
import os

# Ruta a la base de datos
db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'SARA.db')

def hash_password(password):
    """Hashea una contraseña."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(stored_password, provided_password):
    """Verifica una contraseña hasheada."""
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

def register_user(email, password, nombre=None, telefono=None, fecha_nacimiento=None):
    """Registra un nuevo usuario en la base de datos."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Verificar si el usuario ya existe
        cursor.execute("SELECT id FROM usuarios WHERE email = ?", (email,))
        if cursor.fetchone():
            return None  # Usuario ya existe

        hashed_password = hash_password(password)

        cursor.execute("""
            INSERT INTO usuarios (email, password, nombre, telefono, fecha_nacimiento)
            VALUES (?, ?, ?, ?, ?)
        """, (email, hashed_password, nombre, telefono, fecha_nacimiento))

        conn.commit()
        user_id = cursor.lastrowid
        return (user_id,) # Devolver como tupla para consistencia con el route
    except sqlite3.Error as e:
        print(f"Error de base de datos: {e}")
        return None
    finally:
        if conn:
            conn.close()

def login_user(email, password):
    """Autentica a un usuario."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, password FROM usuarios WHERE email = ?", (email,))
        user_row = cursor.fetchone()

        if user_row and verify_password(user_row[1], password):
            return (user_row[0],) # Devolver como tupla
        else:
            return None
    except sqlite3.Error as e:
        print(f"Error de base de datos: {e}")
        return None
    finally:
        if conn:
            conn.close()

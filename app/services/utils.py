import os
import hashlib
import uuid
from dotenv import load_dotenv
from app.constants import ID_PREFIX

load_dotenv()  # Carga .env automáticamente

def get_env(name: str, default=None):
    """
    Obtiene una variable de entorno o lanza un error si no existe.
    """
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"❌ Variable de entorno no encontrada: {name}")
    return value

def hash_text(text: str) -> str:
    """
    Genera un hash único (SHA-256) para un texto.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def generate_uuid() -> str:
    """
    Genera un identificador único
    """
    return f"{ID_PREFIX}{uuid.uuid4().hex[:6].upper()}"
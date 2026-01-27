import os
import hashlib
from dotenv import load_dotenv

load_dotenv()  # Carga .env automáticamente

# Obtiene una variable de entorno o lanza un error si no existe.
def get_env(name: str, default=None):
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"❌ Variable de entorno no encontrada: {name}")
    return value

# Genera un hash único para un texto (para identificar chunks).
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

from pathlib import Path

# ==========================================================
# Define rutas y parámetros comunes para LLM, RAG y almacenamiento.
# ==========================================================

# === Rutas base ===

# Ruta absoluta a la raíz del repositorio
ROOT_DIR = Path(__file__).resolve().parent

# Carpeta de bases vectoriales persistentes (ChromaDB)
CHROMA_PATH = ROOT_DIR / "chroma_db"

# Carpeta de aplicación
APP_PATH = ROOT_DIR / "app"

# Carpeta de documentos
DOCUMENTS_DIR = ROOT_DIR / "app" / "documents"

# === Configuración técnica ===

# Nombre de la colección de documentos en la base de datos
COLLECTION_NAME = "document_collection"

# Modelos usado en la aplicación
EMBEDDING_MODEL = "text-embedding-3-large"
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"

# Configuración del retriever
SEARCH_TYPE = "mmr" # Tipo de búsqueda: 'similarity' o 'mmr'
MMR_DIVERSITY_LAMBDA = 0.7 # Parámetro de diversidad para MMR
MMR_FETCH_K = 20 # Número de documentos a recuperar antes de aplicar MMR
SEARCH_K = 4 # Número de documentos finales a devolver

# Configuracion alternativa para retriever hibrido
ENABLE_HYBRID_SEARCH = True # Habilitar búsqueda híbrida (vectorial + palabras clave)
SIMILARITY_THRESHOLD = 0.70 # Umbral de similitud para incluir documentos en la búsqueda híbrida

# Modelos LLMs default 
DEFAULT_LLM_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
FALLBACK_LLM_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
OPENAI_LLM_MODEL = "gpt-4o-mini"
GOOGLE_LLM_MODEL = "gemini-2.5-flash"
GROQ_LLM_MODEL = "openai/gpt-oss-120b"
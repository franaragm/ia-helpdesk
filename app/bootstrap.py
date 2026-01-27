from config_base import COLLECTION_NAME
from .loader import load_documents
from .vectorstore import create_vectorstore, get_vectorstore

def init_chroma() -> None:
    """
    Inicializa / actualiza ChromaDB de forma incremental.

    - Carga documentos desde /documents
    - Indexa SOLO los nuevos
    - Mantiene los ya existentes
    - Seguro ejecutar múltiples veces
    """
    vectorstore = get_vectorstore()
    count_before = vectorstore._collection.count()

    print(f"📦 Documentos indexados actualmente: {count_before}")

    docs = load_documents()

    if not docs:
        print("⚠️ No se encontraron documentos para indexar.")
        return

    print("🧠 Indexando documentos nuevos (si existen)...")
    create_vectorstore(docs)

    count_after = vectorstore._collection.count()

    if count_after > count_before:
        print(f"✅ Se añadieron {count_after - count_before} nuevos chunks.")
    else:
        print("ℹ️ No había documentos nuevos para indexar.")

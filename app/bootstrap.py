from .loader import load_documents
from .vectorstore import create_vectorstore, get_vectorstore

def init_chroma() -> None:
    """
    Inicializa o actualiza ChromaDB de forma incremental.

    Flujo:
    1) Obtiene el vectorstore persistido (o lo crea si no existe)
    2) Mide cuántos documentos hay actualmente indexados
    3) Carga documentos del directorio /documents
    4) Filtra y añade SOLO los nuevos documentos
    5) Reporta la cantidad de documentos añadidos

    Seguro ejecutar múltiples veces sin duplicar datos.
    """

    # Obtener el vectorstore cacheado
    vectorstore = get_vectorstore()

    # Contar documentos ya indexados
    count_before = vectorstore._collection.count()
    print(f"📦 Documentos indexados actualmente: {count_before}")

    # Cargar documentos y generar chunks
    docs = load_documents()
    if not docs:
        print("⚠️ No se encontraron documentos para indexar.")
        return

    print("🧠 Indexando documentos nuevos (si existen)...")

    # Indexar solo los documentos nuevos (evita duplicados)
    create_vectorstore(docs)

    # Contar nuevamente después de la indexación
    count_after = vectorstore._collection.count()

    # Resultado final
    if count_after > count_before:
        print(f"✅ Se añadieron {count_after - count_before} nuevos chunks.")
    else:
        print("ℹ️ No había documentos nuevos para indexar.")

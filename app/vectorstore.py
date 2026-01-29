from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from app.services.utils import hash_text
import streamlit as st

from config_base import CHROMA_PATH, EMBEDDING_MODEL, COLLECTION_NAME

@st.cache_resource
def get_vectorstore() -> Chroma:
    """
    Devuelve el vectorstore persistido (o lo crea si no existe).
    Se cachea para evitar reconexiones repetidas.
    """
    return Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME
    )

def create_vectorstore(chunks: List[Document]) -> None:
    """
    Indexa chunks NUEVOS en ChromaDB.
    - Mantiene los chunks existentes
    - Evita duplicados usando IDs hash
    """
    
    vectorstore = get_vectorstore()
    
    # Generar IDs únicos por chunk usando el contenido + metadatos
    ids = [
        hash_text(chunk.page_content + str(chunk.metadata))
        for chunk in chunks
    ]
    
    # Obtener IDs ya existentes en la colección
    existing = set(vectorstore.get()["ids"])
    
    # Filtrar solo documentos nuevos
    new_chunks = []
    new_ids = []
    
    for chunk, chunk_id in zip(chunks, ids):
        if chunk_id not in existing:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if not new_chunks:
        print("📦 No hay chunks nuevos para indexar.")
        return

    vectorstore.add_documents(
        documents=new_chunks,
        ids=new_ids,
    )

    print(f"✅ Se indexaron {len(new_chunks)} nuevos chunks en Chroma.")


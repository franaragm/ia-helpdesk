from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config_base import CHROMA_PATH, EMBEDDING_MODEL, OPENAI_LLM_MODEL, COLLECTION_NAME
from .services.utils import hash_text
import streamlit as st

LLM_MODEL = OPENAI_LLM_MODEL

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

def create_vectorstore(docs: List[Document]) -> None:
    """
    Indexa documentos NUEVOS en ChromaDB.
    - Mantiene los documentos existentes
    - Evita duplicados usando IDs hash
    """
    
    vectorstore = get_vectorstore()
    
    # Generar IDs únicos por chunk usando el contenido + metadatos
    ids = [
        hash_text(doc.page_content + str(doc.metadata))
        for doc in docs
    ]
    
    # Obtener IDs ya existentes en la colección
    existing = set(vectorstore.get()["ids"])
    
    # Filtrar solo documentos nuevos
    new_docs = []
    new_ids = []
    
    for doc, doc_id in zip(docs, ids):
        if doc_id not in existing:
            new_docs.append(doc)
            new_ids.append(doc_id)

    if not new_docs:
        print("📦 No hay documentos nuevos para indexar.")
        return

    vectorstore.add_documents(
        documents=new_docs,
        ids=new_ids,
    )

    print(f"✅ Se indexaron {len(new_docs)} nuevos chunks en Chroma.")



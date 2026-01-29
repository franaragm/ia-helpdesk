import streamlit as st
from typing import List
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.services.utils import hash_text
from config_base import DOCUMENTS_DIR

# ===============================
# helpers privados
# ===============================

@st.cache_resource
def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Splitter único y cacheado para todo el sistema (consistencia total)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

def _get_doc_type(filename: str) -> str:
    """Determina el tipo de documento basado en el nombre."""
    if "faq" in filename.lower():
        return "faq"
    elif "manual" in filename.lower():
        return "manual"
    elif "troubleshooting" in filename.lower():
        return "troubleshooting"
    else:
        return "general"
    

# ===============================
# funciones públicas
# ===============================

def load_documents() -> List[Document]:
    """
    Carga documentos markdown del directorio de documentos,
    enriquece metadatos de cada documento, y los divide en chunks.
    returns: Lista de chunks (Document).
    """
    print(f"📚 Cargando documentos desde {DOCUMENTS_DIR}")

    loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    # Enriquecer metadatos para cada documento
    for doc in documents:
        filename = Path(doc.metadata["source"]).stem
        doc.metadata.update({
            "filename": filename,
            "doc_type": _get_doc_type(filename),
            "doc_id": hash_text(doc.page_content)
        })

    print(f"✅ Cargados {len(documents)} documentos desde {DOCUMENTS_DIR}")
    
    print("✂️  Dividiendo documentos en chunks...")
    
    splitter = _get_text_splitter()
    chunks = splitter.split_documents(documents)
    
    print(f"✅ Creados {len(chunks)} chunks")
    
    return chunks

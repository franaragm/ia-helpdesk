import streamlit as st
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile
from .vectorstore import create_vectorstore
from config_base import ROOT_DIR

DOCUMENTS_DIR = ROOT_DIR / "app" / "documents"

@st.cache_resource
def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Splitter único y cacheado para todo el sistema (consistencia total)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

def load_documents(directory: Path = DOCUMENTS_DIR) -> List[Document]:
    """
    Carga PDFs de un directorio y los divide en chunks.
    Usado para bootstrap inicial.
    """
    loader = PyPDFDirectoryLoader(str(directory))
    documents = loader.load()
    print(f"Se cargaron {len(documents)} documentos desde {directory}")

    splitter = _get_text_splitter()
    docs_split = splitter.split_documents(documents)

    print(f"🧠 Se crearon {len(docs_split)} chunks de texto")
    return docs_split

def _load_single_pdf(path: Path, source_name: str) -> List[Document]:
    """
    Carga un único PDF (upload) y lo divide en chunks.
    """
    loader = PyPDFLoader(str(path))
    documents = loader.load()

    splitter = _get_text_splitter()
    docs_split = splitter.split_documents(documents)

    # Metadata consistente para trazabilidad
    for doc in docs_split:
        doc.metadata["source"] = source_name

    print(f"📄 PDF '{source_name}' indexado ({len(docs_split)} chunks)")
    return docs_split

def index_uploaded_pdf(uploaded_file) -> int:
    """
    Indexa incrementalmente un PDF subido por el usuario.
    """

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    docs = _load_single_pdf(tmp_path, uploaded_file.name)

    create_vectorstore(docs)

    return len(docs)

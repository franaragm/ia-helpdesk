from typing import List
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cliente LLM para generación de respuestas
from .services.llm_client import llm_chain_openai

# Configuración global del sistema
from config_base import (
    GENERATION_MODEL,       # Modelo LLM usado para generar la respuesta final
    SEARCH_K,               # Nº máximo de fragmentos que se muestran en la UI
    ENABLE_HYBRID_SEARCH,   # Indica si el retriever es híbrido
    SIMILARITY_THRESHOLD,   # Umbral de similitud (solo informativo para la UI)
)

# Retriever ya construido (MMR + MultiQuery + opcional Hybrid)
from .retrievers import build_retriever

# Prompt principal RAG (contexto + pregunta → respuesta)
from .prompts import rag_prompt

# Modelos Pydantic para respuestas estructuradas
from .schemas import RetrievedDocument, RagResponse

# streamlit para caching
import streamlit as st


# ======================================================
# FORMATEO DE DOCUMENTOS PARA EL PROMPT RAG
# ------------------------------------------------------
# Convierte documentos recuperados en texto legible
# para el LLM, añadiendo:
# - Número de fragmento
# - Fuente (PDF)
# - Página
# ======================================================
def _format_docs(docs) -> str:
    """Formatea los documentos recuperados para el prompt RAG."""
    formatted = []

    for i, doc in enumerate(docs, 1):
        # Cabecera del fragmento
        header = f"[Fragmento {i}]"

        # Metadatos opcionales (fuente y página)
        if doc.metadata:
            if "source" in doc.metadata:
                source = doc.metadata["source"].split("\\")[-1]
                header += f" - Fuente: {source}"
            if "page" in doc.metadata:
                header += f" - Página: {doc.metadata['page']}"

        # Contenido del fragmento
        content = doc.page_content.strip()
        formatted.append(f"{header}\n{content}")

    # Separación clara entre fragmentos
    return "\n\n".join(formatted)


# ======================================================
# CONSTRUCCIÓN DEL PIPELINE RAG
# ------------------------------------------------------
# Flujo:
# Pregunta del usuario
#   → Retriever (MMR + MultiQuery + Hybrid)
#   → Formateo de contexto
#   → Prompt RAG
#   → LLM generador
#   → Texto final
# ======================================================
@st.cache_resource
def build_rag_chain():
    """Construye el pipeline RAG completo."""
    
    # Retriever avanzado (configurado en otro módulo)
    retriever = build_retriever()

    # LLM dedicado EXCLUSIVAMENTE a generar la respuesta final
    llm_generation = llm_chain_openai(
        model=GENERATION_MODEL,
        temperature=0,  # respuestas consistentes y deterministas
    )

    # Definición declarativa del pipeline RAG
    rag_chain = (
        {
            # El retriever recibe la pregunta y produce el contexto
            "context": retriever | _format_docs,

            # La pregunta pasa directamente al prompt
            "question": RunnablePassthrough(),
        }
        | rag_prompt          # Inserta contexto + pregunta en el prompt
        | llm_generation      # Genera la respuesta
        | StrOutputParser()   # Devuelve solo texto plano
    )

    return rag_chain, retriever


# ======================================================
# EJECUCIÓN DE UNA CONSULTA RAG
# ------------------------------------------------------
# Devuelve:
# - Respuesta generada por el LLM
# - Fragmentos usados como soporte (para UI / trazabilidad)
# ======================================================
def query_rag(question: str) -> RagResponse:
    """Ejecuta una consulta RAG y devuelve respuesta + documentos relevantes."""

    # Construir pipeline y retriever
    rag_chain, retriever = build_rag_chain()

    # 1) Generar respuesta usando el pipeline completo
    answer = rag_chain.invoke(question)

    # 2) Recuperar documentos para mostrarlos en la interfaz
    # (se hace por separado para control y transparencia)
    docs = retriever.invoke(question)

    # Convertir documentos LangChain → modelos Pydantic
    documents: List[RetrievedDocument] = []
    for i, doc in enumerate(docs[:SEARCH_K], 1):
        documents.append(
            RetrievedDocument(
                fragmento=i,
                contenido=(
                    doc.page_content[:1000] + "..."
                    if len(doc.page_content) > 1000
                    else doc.page_content
                ),
                fuente=doc.metadata.get("source", "No especificada").split("\\")[-1],
                pagina=doc.metadata.get("page"),
            )
        )

    # Respuesta final estructurada
    return RagResponse(
        answer=answer,
        documents=documents,
    )


# ======================================================
# INFORMACIÓN DEL RETRIEVER PARA LA UI
# ------------------------------------------------------
# Se usa solo para mostrar configuración activa al usuario
# ======================================================
def get_retriever_info() -> dict:
    """Devuelve información del retriever para la UI."""
    return {
        "tipo": "MMR + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": SEARCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A",
    }

from typing import List
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from config_base import GENERATION_MODEL, SEARCH_K
from .services.llm_client import llm_chain_openai
from .retrievers import build_retriever
from .prompts import rag_prompt

# ======================================================
# Funciones auxiliares (NO usan LLM)
# ======================================================

def format_context(docs: List[Document]) -> str:
    """
    Convierte los documentos recuperados por el retriever
    en un bloque de texto limpio y legible para el prompt RAG.

    Responsabilidades:
    - Limitar el número de documentos usados (SEARCH_K)
    - Ignorar fragmentos vacíos
    - Añadir encabezados y fuente para trazabilidad
    - Devolver un único string listo para el prompt
    """
    parts = []

    for i, doc in enumerate(docs[:SEARCH_K], 1):
        content = doc.page_content.strip()

        # Ignorar documentos sin contenido útil
        if not content:
            continue

        # Encabezado del fragmento
        header = f"[Document {i}]"

        # Fuente opcional (filename)
        filename = doc.metadata.get("filename")
        if filename:
            header += f" - Source: {filename}"

        parts.append(f"{header}\n{content}")

    # El prompt RAG recibirá este texto como {contexto}
    return "\n\n".join(parts)


def compute_confidence(rag_answer: str | None, docs: List[Document]) -> float:
    """
    Calcula una confianza heurística simple basada en:
    - Si hay documentos
    - Si RAG sugiere que no puede responder
    Devuelve un número entre 0 y 1.
    """

    if not docs:
        return 0.0  # sin documentos, confianza mínima

    if not rag_answer:
        return 0.2  # documentos pero RAG no generó respuesta

    answer_lower = rag_answer.lower()

    # Penalización si RAG indica no saber / no hay información
    if "no contiene información" in answer_lower or \
       "no se encontró información" in answer_lower or \
       "no incluye información" in answer_lower:
        return 0.3  # confianza baja

    # Si RAG genera algo, confianza moderada-alta
    return 0.6  # valor base para respuestas RAG


def extract_sources(docs: List[Document]) -> List[str]:
    """
    Extrae una lista de fuentes únicas (filename)
    a partir de los documentos recuperados.

    Se usa para:
    - Mostrar trazabilidad en la UI
    - Dar transparencia al usuario
    """
    sources = []

    for doc in docs[:SEARCH_K]:
        filename = doc.metadata.get("filename")
        if filename and filename not in sources:
            sources.append(filename)

    return sources


# ======================================================
# Construcción del pipeline RAG (LCEL)
# ======================================================

@st.cache_resource
def build_rag_chain():
    """
    Construye el pipeline RAG completo usando LCEL.

    Flujo declarativo:
    Pregunta del usuario
      → Retriever (MMR + MultiQuery + opcional híbrido)
      → Formateo del contexto
      → Prompt RAG
      → LLM generador
      → Texto plano como salida
    """

    # Retriever avanzado (vectorstore + estrategias)
    retriever = build_retriever()

    # LLM usado SOLO para generar la respuesta final
    llm_generation = llm_chain_openai(
        model=GENERATION_MODEL,
        temperature=0,  # respuestas deterministas
    )

    # Pipeline LCEL
    rag_chain = (
        {
            # El retriever recibe la consulta y produce contexto
            "context": retriever | format_context,

            # La consulta pasa directamente al prompt
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, retriever


# ======================================================
# API pública del módulo RAG
# ======================================================

def query_rag(query: str) -> dict:
    """
    Ejecuta una consulta RAG completa y devuelve un objeto dict con toda la información.

    Flujo:
    1) Recupera documentos relevantes con el retriever (MMR + MultiQuery ± Hybrid)
    2) Formatea el contexto para el prompt RAG
    3) Genera la respuesta con el LLM
    4) Calcula confianza y extrae fuentes
    5) Devuelve un dict con toda la información

    Casos especiales manejados:
    - Sin documentos encontrados → respuesta de error con baja confianza
    - Documentos sin contenido útil → mensaje de advertencia con confianza intermedia
    """
    
    # Construir pipeline y retriever
    rag_chain, retriever = build_rag_chain()

    # Recuperar documentos (también se usan para scoring y fuentes)
    docs: List[Document] = retriever.invoke(query)

    # ==========================================
    # Caso 1: No se recuperó ningún documento
    # ==========================================
    if not docs:
        return {
            "answer": "No se encontró información relevante en la base de conocimiento.",
            "confidence": 0.1,
            "sources": [],
            "rag_context": "",
            "requires_human": True,
        }

    # Formatear contexto y extraer fuentes
    context = format_context(docs)
    sources = extract_sources(docs)

    # ==========================================
    # Caso 2: Documentos existen pero sin contenido útil
    # ==========================================
    if not context.strip():
        return {
            "answer": "Se encontraron documentos, pero no contienen información útil.",
            "confidence": 0.2,
            "sources": sources,
        }

    # ==========================================
    # Caso normal: ejecutar pipeline RAG
    # ==========================================
    answer = rag_chain.invoke(query)
    confidence = compute_confidence(answer, docs)

    # Devolver respuesta
    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources
    }

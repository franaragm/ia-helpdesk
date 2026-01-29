from typing import List
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from config_base import GENERATION_MODEL, SEARCH_K
from .services.llm_client import llm_chain_openai
from .retrievers import build_retriever
from .prompts import rag_prompt
from .schemas import RagAnswer

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


def compute_confidence(query: str, docs: List[Document]) -> float:
    """
    Calcula un nivel de confianza heurístico (NO basado en LLM).

    La confianza se construye combinando tres señales simples:
    1) Coincidencia de palabras clave entre la consulta y los documentos
    2) Número de documentos recuperados
    3) Cantidad total de texto disponible como contexto

    El objetivo NO es precisión matemática,
    sino dar una estimación razonable y estable para la UI.
    """

    # Sin documentos → confianza mínima
    if not docs:
        return 0.0

    # Extraer palabras clave de la consulta
    # (se ignoran palabras muy cortas)
    keywords = {w for w in query.lower().split() if len(w) > 2}

    # Si no hay keywords claras, devolver un valor base neutro
    if not keywords:
        return 0.3

    keyword_matches = 0
    total_words = 0

    # Analizamos solo los primeros documentos (los más relevantes)
    for doc in docs[:3]:
        content = doc.page_content.lower()
        words = content.split()

        total_words += len(words)

        # Contar cuántas keywords aparecen en el contenido
        keyword_matches += sum(1 for k in keywords if k in content)

    # --- Cálculo de la puntuación ---

    # Base: proporción de keywords encontradas (máx 1.0)
    base_score = min(keyword_matches / len(keywords), 1.0)

    # Bonus por número de documentos (máx +0.2)
    docs_bonus = min(len(docs) / 4.0, 0.2)

    # Bonus por tamaño del contexto (máx +0.1)
    length_bonus = min(total_words / 1000.0, 0.1)

    # Resultado final limitado a 1.0
    confidence = base_score + docs_bonus + length_bonus

    return round(min(confidence, 1.0), 2)


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

def query_rag(query: str) -> RagAnswer:
    """
    Ejecuta una consulta RAG completa y devuelve un objeto Pydantic RagAnswer.

    Flujo:
    1) Recupera documentos relevantes con el retriever (MMR + MultiQuery ± Hybrid)
    2) Formatea el contexto para el prompt RAG
    3) Genera la respuesta con el LLM
    4) Calcula confianza y extrae fuentes
    5) Devuelve un RagAnswer con toda la información

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
        return RagAnswer(
            answer="No se encontró información relevante en la base de conocimiento.",
            confidence=0.1,
            sources=[],
        )

    # Formatear contexto y extraer fuentes
    context = format_context(docs)
    sources = extract_sources(docs)

    # ==========================================
    # Caso 2: Documentos existen pero sin contenido útil
    # ==========================================
    if not context.strip():
        return RagAnswer(
            answer="Se encontraron documentos, pero no contienen información útil.",
            confidence=0.2,
            sources=sources,
        )

    # ==========================================
    # Caso normal: ejecutar pipeline RAG
    # ==========================================
    answer = rag_chain.invoke(query)
    confidence = compute_confidence(query, docs)

    # Devolver respuesta tipada
    return RagAnswer(
        answer=answer,
        confidence=confidence,
        sources=sources,
    )

from typing import List, Tuple
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from config_base import GENERATION_MODEL, SEARCH_K
from .services.llm_client import llm_chain_openai
from .constants import RAG_NEGATIVE_PHRASES, STOPWORDS
from .vectorstore import get_vectorstore
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


def compute_confidence(query: str, rag_answer: str | None, docs: List[Document], scored_docs: List[Tuple[Document, float]]) -> float:
    """
    Calcula una confianza heurística entre 0 y 1 basada en:
    - Presencia de documentos
    - Scores de similitud del vectorstore
    - Calidad y longitud de la respuesta RAG
    - Coincidencia parcial de la query en la respuesta
    - Señales explícitas de "no sabe / no información"
    """

    # =========================
    # Guard rails duros
    # =========================
    if not docs or not rag_answer:
        return 0.0  # sin documentos o sin respuesta

    answer_lower = rag_answer.lower()
    query_lower = query.lower()

    # =========================
    # Penalización fuerte si el modelo dice que no sabe
    # =========================
    if any(phrase in answer_lower for phrase in RAG_NEGATIVE_PHRASES):
        return 0.2  # muy baja confianza

    # =========================
    # Confianza base
    # =========================
    confidence = 0.4
    
    # =========================
    # Señal de retrieval (SCORES)
    # =========================
    if scored_docs:
        distances = [score for _, score in scored_docs]

        # Normalizar distancia → similitud (0..1]
        similarities = [1 / (1 + d) for d in distances]

        avg_similarity = sum(similarities) / len(similarities)

        # Peso fuerte: retrieval es clave
        confidence += 0.4 * avg_similarity

    # =========================
    # Cantidad de documentos útiles
    # =========================
    if len(docs) >= 5:
        confidence += 0.1
    elif len(docs) >= 3:
        confidence += 0.05

    # =========================
    # Longitud de la respuesta
    # =========================
    answer_len = len(rag_answer.split())

    if answer_len > 50:
        confidence += 0.05
    elif answer_len < 10:
        confidence -= 0.1

    # =========================
    # Coincidencia léxica query ↔ respuesta
    # =========================
    query_words = [
        w for w in query_lower.split()
        if w not in STOPWORDS
    ]

    if query_words:
        matches = sum(1 for w in query_words if w in answer_lower)
        match_ratio = matches / len(query_words)
        confidence += 0.2 * match_ratio

    # =========================
    # Clamp final
    # =========================
    return min(max(confidence, 0.0), 1.0)


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


def empty_rag_response(message: str, sources=None) -> dict:
    """
    Devuelve una respuesta RAG vacía con un mensaje específico.
    Utilizado en casos donde no se pueden recuperar documentos relevantes.
    """
    return {
        "answer": message,
        "confidence": 0.0,
        "sources": sources or [],
        "rag_context": "",
        "requires_human": True,
    }



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
        return empty_rag_response(
            "No se encontró información relevante en la base de conocimiento."
        )

    # Formatear contexto y extraer fuentes
    context = format_context(docs)
    sources = extract_sources(docs)

    # ==========================================
    # Caso 2: Documentos existen pero sin contenido útil
    # ==========================================
    if not context.strip():
        return empty_rag_response(
            "Se encontraron documentos, pero no contienen información útil.",
            sources=sources
        )

    # ==========================================
    # Caso normal: ejecutar pipeline RAG
    # ==========================================
    answer = rag_chain.invoke(query)
    
    # Acceso directo al vectorstore para scoring (opcional)
    vectorstore = get_vectorstore()
    scored_docs = vectorstore.similarity_search_with_score(query, k=SEARCH_K)
    
    # Calcular confianza pieza clave del sistema para posterior clasificación
    confidence = compute_confidence(query, answer, docs, scored_docs)

    # Devolver respuesta
    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources
    }

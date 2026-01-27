from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever

# Configuración global del sistema RAG
from config_base import (
    SEARCH_TYPE,              # Tipo de búsqueda base (ej: "mmr")
    SEARCH_K,                 # Número final de fragmentos a recuperar
    ENABLE_HYBRID_SEARCH,     # Activa combinación de estrategias de búsqueda
    SIMILARITY_THRESHOLD,     # Umbral mínimo de similitud en modo híbrido
    MMR_DIVERSITY_LAMBDA,     # Controla balance relevancia vs diversidad en MMR
    MMR_FETCH_K,              # Número de candidatos iniciales para MMR
    QUERY_MODEL,              # Modelo LLM usado para generar consultas alternativas
)

from .vectorstore import get_vectorstore
from .services.llm_client import llm_chain_openai
from .prompts import multi_query_prompt
import streamlit as st

@st.cache_resource
def build_retriever() -> BaseRetriever:
    """
    Construye y devuelve el retriever principal del sistema RAG.

    Estrategia utilizada:
    1) MMR Retriever → evita fragmentos redundantes
    2) MultiQuery Retriever → reformula la pregunta para mejorar el recall
    3) (Opcional) Ensemble Retriever → combina MultiQuery+MMR con Similarity

    El resultado es un retriever robusto, equilibrado y tolerante
    a preguntas mal formuladas o incompletas.
    """

    # === Acceso al vectorstore persistido (ChromaDB) ===
    vectorstore = get_vectorstore()

    # === LLM dedicado EXCLUSIVAMENTE a generar variantes de la consulta ===
    # No se usa para responder, solo para reformular preguntas
    llm_queries = llm_chain_openai(
        model=QUERY_MODEL,
        temperature=0,  # determinista: mismas queries para misma pregunta
    )

    # ======================================================
    # 1) RETRIEVER BASE: MMR (Maximal Marginal Relevance)
    # ------------------------------------------------------
    # Objetivo:
    # - Recuperar fragmentos relevantes
    # - Evitar fragmentos muy similares entre sí
    #
    # fetch_k → candidatos iniciales
    # k       → fragmentos finales
    # lambda  → equilibrio relevancia / diversidad
    # ======================================================
    base_retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,  # normalmente "mmr"
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K,
        },
    )

    # ======================================================
    # 2) RETRIEVER DE SIMILARITY (búsqueda directa)
    # ------------------------------------------------------
    # Objetivo:
    # - Capturar coincidencias exactas o muy cercanas
    # - Actúa como complemento "preciso" al enfoque exploratorio
    # ======================================================
    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_K},
    )

    # ======================================================
    # 3) MULTIQUERY RETRIEVER (sobre MMR)
    # ------------------------------------------------------
    # Objetivo:
    # - Generar múltiples versiones de la pregunta del usuario
    # - Ejecutar MMR para cada variante
    # - Unir resultados y eliminar duplicados
    #
    # Beneficio:
    # - Aumenta el recall
    # - Reduce dependencia de una única formulación
    # ======================================================
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,   # MMR como base sólida
        llm=llm_queries,            # LLM para generar variantes
        prompt=multi_query_prompt, # Prompt legal especializado
    )

    # ======================================================
    # 4) ENSEMBLE RETRIEVER (opcional)
    # ------------------------------------------------------
    # Combina:
    # - Exploración semántica (MultiQuery + MMR)
    # - Precisión directa (Similarity)
    #
    # weights:
    # - 70% exploración
    # - 30% precisión
    # ======================================================
    if ENABLE_HYBRID_SEARCH:
        return EnsembleRetriever(
            retrievers=[
                mmr_multi_retriever,
                similarity_retriever,
            ],
            weights=[0.7, 0.3],
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

    # === Modo no híbrido: solo MultiQuery + MMR ===
    return mmr_multi_retriever

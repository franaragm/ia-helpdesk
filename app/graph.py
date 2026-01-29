from typing import Dict
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

from config_base import OPENAI_LLM_MODEL
from .schemas import HelpdeskState
from .rag import query_rag
from .prompts import classification_prompt
from .services.llm_client import llm_chain_openai


# ======================================================
# NODO 1: EJECUTAR RAG
# ======================================================

def run_rag(state: HelpdeskState) -> Dict:
    """
    Ejecuta el sistema RAG sobre la consulta del usuario
    y guarda el resultado en el estado.
    """
    result = query_rag(state["query"])

    return {
        "rag_answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"],
        "rag_context": result.get("rag_context", result["answer"]),
        "history": [
            "RAG ejecutado con MultiQuery + MMR",
            f"Confianza heurística obtenida: {result['confidence']}",
            f"Fuentes consultadas: {len(result['sources'])}",
        ],
    }

# ======================================================
# NODO 2: CLASIFICAR (AUTOMÁTICO VS ESCALADO)
# ======================================================

def classify_with_context(state: HelpdeskState) -> Dict:
    """
    Decide si la respuesta puede entregarse automáticamente
    o si debe escalarse a un humano.
    """

    llm = llm_chain_openai(
        model=OPENAI_LLM_MODEL,
        temperature=0.1,
    )

    response = llm.invoke(
        classification_prompt.format(
            question=state["query"],
            context=state.get("rag_context", ""),
            confidence=state.get("confidence", 0),
        )
    )

    content = response.content.lower()

    if "automatic" in content:
        category = "automatic"
    elif "escalated" in content:
        category = "escalated"
    else:
        # fallback defensivo basado en confianza
        category = "automatic" if state.get("confidence", 0) >= 0.6 else "escalated"

    return {
        "category": category,
        "history": [
            f"Clasificación realizada: {category}",
            f"Justificación LLM: {response.content}",
        ],
    }


# ======================================================
# NODO 3: PREPARAR ESCALADO
# ======================================================

def prepare_escalation(state: HelpdeskState) -> Dict:
    """
    Marca el estado como pendiente de intervención humana.
    Inicializa human_answer vacío para que LangGraph no rompa.
    """
    return {
        "requires_human": True,
        "human_answer": None,
        "history": ["Consulta escalada a agente humano."],
    }


# ======================================================
# NODO 4: PROCESAR RESPUESTA HUMANA
# ======================================================

def process_human_answer(state: HelpdeskState) -> Dict:
    """
    Si el agente humano ya respondió, se fija como respuesta final.
    """
    if state.get("human_answer"):
        return {
            "final_answer": state["human_answer"],
            "history": ["Respuesta proporcionada por agente humano."],
        }

    return {
        "history": ["Esperando respuesta del agente humano."],
    }


# ======================================================
# NODO 5: GENERAR RESPUESTA FINAL
# ======================================================

def generate_final_answer(state: HelpdeskState) -> Dict:
    """
    Genera la respuesta final para el usuario.
    Si no hay humano, se usa la respuesta del RAG.
    """
    if state.get("final_answer"):
        return {
            "history": ["Respuesta final ya establecida por humano."],
        }

    answer = state.get("rag_answer", "")
    sources = state.get("sources", [])

    if sources:
        answer += "\n\nFuentes consultadas: " + ", ".join(sources)

    return {
        "final_answer": answer,
        "history": ["Respuesta final generada automáticamente."],
    }


# ======================================================
# FUNCIONES DE ENRUTAMIENTO
# ======================================================

def route_after_classification(state: HelpdeskState) -> str:
    """
    Decide el siguiente nodo tras la clasificación.
    """
    return "final_answer" if state["category"] == "automatic" else "escalation"


# ======================================================
# CREACIÓN Y COMPILACIÓN DEL GRAFO
# ======================================================

def build_helpdesk_graph():
    """
    Construye el grafo de LangGraph con todos los nodos y transiciones.
    """

    graph = StateGraph(HelpdeskState)

    graph.add_node("rag", run_rag)
    graph.add_node("classify", classify_with_context)
    graph.add_node("escalation", prepare_escalation)
    graph.add_node("process_human", process_human_answer)
    graph.add_node("final_answer", generate_final_answer)

    graph.add_edge(START, "rag")
    graph.add_edge("rag", "classify")

    graph.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "final_answer": "final_answer",
            "escalation": "escalation",
        },
    )

    graph.add_edge("escalation", "process_human")
    graph.add_edge("process_human", END)
    graph.add_edge("final_answer", END)

    return graph


def compile_helpdesk():
    """
    Compila el grafo con checkpoint persistente en SQLite.
    Permite interrumpir y reanudar ejecuciones (human-in-the-loop).
    """
    conn = sqlite3.connect("helpdesk.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = build_helpdesk_graph()

    # Cuando el flujo vaya a ejecutar el nodo `process_human`,
    # la ejecución se interrumpe ANTES, se persiste el estado
    # y queda pendiente hasta que un agente humano lo reanude.
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["process_human"],
    )

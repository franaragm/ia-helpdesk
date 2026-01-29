from typing import TypedDict, Optional, List, Annotated
from operator import add
from pydantic import BaseModel, Field

# ======================================================
# ESQUEMA Pydantic para VALIDACIÓN DE DATOS DEL GRAFO
# ======================================================
    
class HelpdeskStateModel(BaseModel):
    """
    Versión Pydantic del estado de Helpdesk.
    Sirve para validar que final_state.values cumple la estructura esperada.
    """
    query: str
    rag_answer: Optional[str]
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[str] = []
    rag_context: Optional[str]
    category: Optional[str]
    requires_human: bool
    human_answer: Optional[str]
    final_answer: Optional[str]
    history: List[str] = []

# ======================================================
# ESQUEMA TypedDict para ESTADO MUTABLE DEL GRAFO
# ======================================================

# Usar TypedDict para el estado mutable del grafo no para validar,
# sino para facilitar la lectura y autocompletado en los nodos.
class HelpdeskState(TypedDict):
    """
    Estado global que se va propagando entre nodos del grafo.
    Cada nodo puede leer y devolver parcialmente este estado.
    """

    query: str

    # Resultado del RAG
    rag_answer: Optional[str]
    confidence: float
    sources: List[str]
    rag_context: Optional[str]

    # Clasificación
    category: Optional[str]  # "automatic" | "escalated"

    # Escalado humano
    requires_human: bool
    human_answer: Optional[str]

    # Respuesta final
    final_answer: Optional[str]

    # Historial explicativo del flujo
    history: Annotated[List[str], add]
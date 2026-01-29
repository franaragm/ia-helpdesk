# 🧠 Visión general del flujo

Tu grafo implementa este **patrón clásico de Helpdesk 2.0**:

```
Usuario
  ↓
RAG (respuesta automática)
  ↓
Clasificación (¿confiamos en la respuesta?)
  ├── Sí → Respuesta final
  └── No → Escalado humano
               ↓
         Agente humano
```

No hay magia. Solo **control explícito del flujo y nodos especializados**.

---

# 1️⃣ Nodos del grafo (qué hace cada uno)

### 🔹 `rag` → `run_rag`

* Ejecuta el pipeline RAG (`query_rag`)
* Busca documentos y genera una respuesta preliminar
* Calcula `confidence` heurística
* **No decide nada**, solo produce información

Ejemplo de estado que genera:

```python
{
  "rag_answer": "...",
  "confidence": 0.62,
  "sources": [...],
  "history": ["RAG ejecutado...", "Confianza heurística...", "Fuentes consultadas..."]
}
```

---

### 🔹 `classify` → `classify_with_context`

* Analiza la salida del RAG
* Decide **qué camino tomar**: `automatic` o `escalated`
* No genera texto para el usuario
* Devuelve la categoría y justificación en `history`

Ejemplo:

```python
{
  "category": "automatic",
  "history": ["Clasificación realizada: automatic", "Justificación LLM: ..."]
}
```

⚠️ Este es el **nodo de decisión principal**.

---

### 🔹 `escalation` → `prepare_escalation`

* Marca el estado como pendiente de intervención humana (`requires_human = True`)
* Añade historial explicativo
* **No decide nada**, solo normaliza el estado antes del handoff

Ejemplo:

```python
{
  "requires_human": True,
  "human_answer": None,
  "history": ["Consulta escalada a agente humano."]
}
```

---

### 🔹 `process_human` → `process_human_answer`

* Gestiona la respuesta del agente humano

* Puede:

  * Esperar input externo
  * Leer una cola
  * Recibir respuesta mock

* Genera `final_answer` con la respuesta humana

Ejemplo:

```python
{
  "final_answer": "Respuesta del agente humano",
  "history": ["Respuesta proporcionada por agente humano."]
}
```

---

### 🔹 `final_answer` → `generate_final_answer`

* Toma la respuesta del RAG si no hay humano
* Añade fuentes y ajusta el formato
* Marca la respuesta final del ticket
* **No toma decisiones**, solo prepara la salida final

---

# 2️⃣ Flujo de edges (cómo se mueve la información)

```python
START → rag → classify
```

* Siempre empieza ejecutando RAG
* Luego `classify` evalúa si se entrega automáticamente o se escala

### Decisión principal

```python
graph.add_conditional_edges(
    "classify",
    route_after_classification,
    {
        "final_answer": "final_answer",
        "escalation": "escalation",
    }
)
```

Función típica:

```python
def route_after_classification(state):
    return "final_answer" if state["category"] == "automatic" else "escalation"
```

---

### Camino A: Respuesta automática

```python
final_answer → END
```

* Flujo corto
* Usuario recibe respuesta inmediata

---

### Camino B: Escalado humano

1. `classify → escalation` → marca `requires_human = True`
2. `escalation → process_human` → espera input humano
3. `process_human → END` → entrega respuesta final

---

# 3️⃣ Checkpointer SQLite

```python
conn = sqlite3.connect("helpdesk.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
```

* Guarda el estado de cada nodo
* Permite reanudar flujos si hay crash o espera humana
* Mantiene `HelpdeskState`, historial y metadata

---

# 4️⃣ Compilación del grafo

```python
graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["process_human"],
)
```

* Convierte el grafo estático en **motor ejecutable**

* `interrupt_before=["process_human"]`:

  * Pausa el flujo antes del nodo humano
  * Guarda el estado en SQLite
  * Permite que la UI humana lo reanude

* Sin esto, `process_human` se ejecutaría automáticamente y rompería la lógica human-in-the-loop

---

# 5️⃣ Resumen mental

> 🔹 **RAG** produce información
> 🔹 **Classify** decide el camino
> 🔹 **Escalation** marca estado
> 🔹 **Router** solo lee flags
> 🔹 **Final / Humano** cierran flujo

Este patrón **hace el flujo legible, testable y escalable**.

---


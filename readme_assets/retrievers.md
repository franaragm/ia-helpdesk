## 🧠 Idea central (antes del detalle)

Tu sistema responde a esta pregunta implícita:

> **“¿Cómo recuperar fragmentos relevantes, variados y confiables para cualquier pregunta del usuario?”**

La respuesta es:

1. **Primero**: recuperar fragmentos relevantes usando MMR
2. **Después**: reformular la pregunta para no perder información (MultiQuery)
3. **Después**: añadir búsqueda directa por similitud (Similarity Retriever)
4. **Opcionalmente**: combinar resultados con pesos (EnsembleRetriever)

Cada retriever resuelve **un problema distinto**.

---

## 1️⃣ MMR Retriever (base sólida)

```python
base_retriever = vectorstore.as_retriever(
    search_type=SEARCH_TYPE,  # normalmente "mmr"
    search_kwargs={
        "k": SEARCH_K,
        "lambda_mult": MMR_DIVERSITY_LAMBDA,
        "fetch_k": MMR_FETCH_K,
    },
)
```

### Qué hace

* Recupera fragmentos relevantes de manera **diversa**
* Evita fragmentos **muy similares entre sí**
* `fetch_k` → candidatos iniciales
* `k` → fragmentos finales
* `lambda_mult` → equilibrio relevancia/diversidad

### Problema que resuelve

❌ Con solo similarity, podrías obtener muchos fragmentos casi idénticos, ignorando información complementaria.

---

## 2️⃣ Similarity Retriever (búsqueda directa)

```python
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": SEARCH_K},
)
```

### Qué hace

* Encuentra coincidencias exactas o muy cercanas a la pregunta
* Actúa como **complemento preciso** al enfoque exploratorio de MMR

### Por qué se conserva

Porque a veces la pregunta está perfectamente formulada y similarity devuelve **exactamente lo que necesitas**.

---

## 3️⃣ MultiQuery Retriever (reformulación inteligente)

```python
mmr_multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm_queries,
    prompt=multiquery_prompt,
)
```

### Problema que resuelve

Las preguntas de los usuarios pueden ser:

* Mal formuladas
* Parciales o ambiguas

MultiQuery:

1. Genera múltiples **variantes de la pregunta** con un LLM
2. Ejecuta MMR para cada variante
3. Une los resultados y elimina duplicados

> Esto **aumenta el recall** sin perder información importante.

---

### Por qué MultiQuery se basa en MMR y no en similarity

* Ejecutar varias búsquedas con similarity produciría **demasiada redundancia**
* MMR filtra cada búsqueda y asegura diversidad

📌 **MMR = base sólida**
📌 **MultiQuery = expansión inteligente**

---

## 4️⃣ Ensemble Retriever (combinación final)

```python
EnsembleRetriever(
    retrievers=[mmr_multi_retriever, similarity_retriever],
    weights=[0.7, 0.3],
    similarity_threshold=SIMILARITY_THRESHOLD,
)
```

### Qué hace

Combina estrategias:

| Estrategia       | Rol                     |
| ---------------- | ----------------------- |
| MultiQuery + MMR | Exploración inteligente |
| Similarity       | Francotirador preciso   |

### Pesos

* 70% confianza en exploración semántica
* 30% confianza en match directo

`similarity_threshold` evita resultados irrelevantes.

---

## 5️⃣ Diagrama mental completo

```
Pregunta del usuario
  │
  ├─ Similarity ───────────────┐
  │                             ├─ Ensemble ─► docs finales
  └─ MultiQuery
        ├─ variante 1 ─► MMR ─┐
        ├─ variante 2 ─► MMR ─┤
        └─ variante 3 ─► MMR ─┘
```

---

## 6️⃣ Resumen ultra claro

| Componente          | Por qué existe                          |
| ------------------- | --------------------------------------- |
| SimilarityRetriever | Precisión directa                       |
| MMRRetriever        | Evita duplicados                        |
| MultiQueryRetriever | No perder información por mala pregunta |
| EnsembleRetriever   | Combina precisión + cobertura           |

---

## 7️⃣ Si quisieras simplificar (opcional)

### Nivel básico

```python
retriever = vectorstore.as_retriever(search_type="similarity", k=3)
```

### Nivel intermedio

```python
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_type="mmr", k=3),
    llm=llm_queries,
)
```

### Nivel avanzado (tu caso actual)

✔ MultiQuery + MMR + (opcional) Similarity + Ensemble

---

## 🏁 Conclusión

No es complejo, es **una estrategia en capas**:

> **Explorar bien → no repetir → no depender de una sola formulación → combinar enfoques**

Esto es **arquitectura RAG robusta y tolerante a preguntas mal formuladas**.

---

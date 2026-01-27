# 🧠 Idea central (antes del detalle)

Tu sistema responde a esta pregunta implícita:

> **“¿Cómo recupero los fragmentos más útiles, variados y robustos posibles para una pregunta legal?”**

La respuesta es:

1. **Primero**: buscar fragmentos relevantes
2. **Después**: evitar fragmentos redundantes
3. **Después**: reformular la pregunta para no perder información
4. **Opcionalmente**: combinar estrategias distintas

Cada retriever resuelve **un problema distinto**.

---

# 1️⃣ VectorStore Retriever (el más básico)

```python
vectorstore.as_retriever(search_type="similarity", k=SEARCH_K)
```

### Qué hace

* Busca los `K` fragmentos **más parecidos semánticamente** a la pregunta.
* Usa distancia de embeddings (coseno, L2, etc.).

### Problema que tiene

❌ Si hay muchos fragmentos parecidos:

* Te devuelve **trozos casi idénticos**
* Ignora otros aspectos relevantes

Ejemplo:

```
Contrato A: “El arrendatario es Juan Pérez…”
Contrato B: “El arrendatario es Juan Pérez…”
Contrato C: “Duración del contrato: 12 meses…”
```

Si preguntas:

> “¿Quién es el arrendatario?”

Similarity puede devolver A y B → **redundancia**

---

# 2️⃣ MMR Retriever (Maximal Marginal Relevance)

```python
base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    k=SEARCH_K,
    fetch_k=MMR_FETCH_K,
    lambda_mult=MMR_DIVERSITY_LAMBDA,
)
```

### Qué hace

MMR responde a:

> “Dame fragmentos relevantes, pero **no repetidos**”

### Cómo funciona

1. Busca `fetch_k` candidatos relevantes
2. Selecciona `k` fragmentos:

   * relevantes **y**
   * diferentes entre sí

### Resultado

* Menos redundancia
* Más cobertura de información

💡 **Por eso este es tu “base_retriever”**
Es una **mejora directa** sobre similarity.

---

# 3️⃣ MultiQueryRetriever (el salto de calidad)

```python
mmr_multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm_queries,
)
```

### Problema que resuelve

Las personas preguntan **mal** o **de forma parcial**.

Ejemplo:

> “¿Quién vive en el piso?”

Pero el contrato dice:

* arrendatario
* inquilino
* parte arrendataria

### Qué hace MultiQuery

1. Usa un LLM para generar **3 versiones alternativas** de la pregunta
2. Ejecuta el retriever (MMR) **para cada versión**
3. Une y deduplica los resultados

Ejemplo:

```
Original: ¿Quién vive en el piso?
Variantes:
- ¿Quién es el arrendatario del inmueble?
- ¿Quién figura como inquilino en el contrato?
- ¿Qué persona ocupa la vivienda?
```

👉 Esto **multiplica la capacidad de recall** (no perder info).

---

### Por qué MultiQuery usa MMR y no similarity

Porque:

* Ya estás ejecutando **varias búsquedas**
* Sin MMR, tendrías **muchísima redundancia**
* MMR filtra mejor cada búsqueda

📌 **MMR = base sólida**
📌 **MultiQuery = expansión inteligente**

---

# 4️⃣ Similarity Retriever (por qué sigue existiendo)

```python
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    k=SEARCH_K,
)
```

### ¿No era malo similarity?

No. Es:

* Muy preciso
* Muy directo
* Muy rápido

Pero:

* Puede ser demasiado estrecho

### Por qué lo conservas

Porque a veces:

* La pregunta está **perfectamente formulada**
* Similarity devuelve el fragmento exacto
* MultiQuery + MMR puede “diluir” eso

---

# 5️⃣ EnsembleRetriever (la combinación final)

```python
EnsembleRetriever(
    retrievers=[mmr_multi_retriever, similarity_retriever],
    weights=[0.7, 0.3],
)
```

### Qué hace

Combina resultados de **distintas estrategias**.

Piénsalo así:

| Estrategia       | Rol                    |
| ---------------- | ---------------------- |
| MultiQuery + MMR | Explorador inteligente |
| Similarity       | Francotirador preciso  |

### Pesos

```python
weights=[0.7, 0.3]
```

* 70% confianza en exploración semántica
* 30% confianza en match directo

### similarity_threshold

Evita meter basura irrelevante.

---

# 6️⃣ Diagrama mental completo

```
Pregunta
  │
  ├─ Similarity ───────────────┐
  │                             ├─ Ensemble ─► docs finales
  └─ MultiQuery
        ├─ variante 1 ─► MMR ─┐
        ├─ variante 2 ─► MMR ─┤
        └─ variante 3 ─► MMR ─┘
```

---

# 7️⃣ Resumen ultra claro

| Componente          | Por qué existe                          |
| ------------------- | --------------------------------------- |
| SimilarityRetriever | Precisión directa                       |
| MMRRetriever        | Evita duplicados                        |
| MultiQueryRetriever | No perder información por mala pregunta |
| EnsembleRetriever   | Combina precisión + cobertura           |

---

# 8️⃣ Si quisieras simplificar (opcional)

### Nivel básico

```python
retriever = vectorstore.as_retriever(search_type="similarity", k=3)
```

### Nivel intermedio (recomendado)

```python
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_type="mmr", k=3),
    llm=llm_queries,
)
```

### Nivel avanzado (tu caso actual)

✔ Exactamente lo que tienes

---

## 🏁 Conclusión

No es un lío, es una **estrategia en capas**:

> **Explorar bien → no repetir → no depender de una sola forma de preguntar → combinar enfoques**

Esto es **arquitectura RAG madura**.

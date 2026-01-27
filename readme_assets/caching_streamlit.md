## 1️⃣ ¿Qué hace realmente `@st.cache_resource`?

`cache_resource` está pensado para **objetos pesados y persistentes**:

* Conexiones
* Clientes
* Modelos
* Vectorstores
* Pipelines RAG completos

👉 Se ejecuta **una sola vez por sesión de Streamlit**, no por cada interacción.

📌 Diferencia clave:

| Decorador           | Para qué                          |
| ------------------- | --------------------------------- |
| `st.cache_data`     | Datos (listas, dicts, DataFrames) |
| `st.cache_resource` | Objetos vivos / costosos          |

---

## 2️⃣ Dónde SÍ usar `cache_resource` en tu proyecto

### 🟢 A) Vectorstore (altamente recomendado)

Ahora mismo, cada vez que preguntas:

```python
vectorstore = get_vectorstore()
```

Eso:

* Reabre Chroma
* Reinstancia embeddings
* Relee metadata

💥 Innecesario

### ✅ Solución ideal

```python
# vectorstore.py
import streamlit as st

@st.cache_resource
def get_vectorstore() -> Chroma:
    return Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
    )
```

✔ Se carga una vez
✔ Se reutiliza en toda la sesión
✔ Mucho más rápido

---

### 🟢 B) Retriever completo

Tu `build_retriever()`:

* Crea MMR
* Crea MultiQuery
* Crea Ensemble

Todo eso **no cambia entre preguntas**.

### ✅ Muy buena candidata

```python
# retrievers.py
import streamlit as st

@st.cache_resource
def build_retriever() -> BaseRetriever:
    ...
```

Beneficio:

* No se reconstruye cada vez
* Menos llamadas al LLM de queries
* Comportamiento estable

---

### 🟢 C) Pipeline RAG completo

Esto:

```python
rag_chain, retriever = build_rag_chain()
```

👉 es **costoso** y **determinista**

### ✅ Ideal para cache_resource

```python
# rag.py
@st.cache_resource
def build_rag_chain():
    ...
```

Luego en `query_rag`:

```python
rag_chain, retriever = build_rag_chain()
```

Streamlit devolverá el mismo objeto ya creado.

---

## 3️⃣ Dónde NO usar `cache_resource`

### 🔴 A) `query_rag()`

❌ NO

```python
@st.cache_resource
def query_rag(...):
```

¿Por qué?

* La entrada (`question`) cambia
* Cachearías respuestas equivocadas
* Memory leak potencial

---

### 🔴 B) Funciones con estado mutable

Ejemplo peligroso:

```python
@st.cache_resource
def get_retriever():
    retriever.some_internal_state.append(...)
```

Si el objeto cambia internamente → cache corrupta.

📌 En tu caso estás bien: los retrievers son **inmutables**.

---

## 4️⃣ Recomendación final para TU proyecto

### 🥇 Nivel óptimo de caching

| Componente          | Decorador            |
| ------------------- | -------------------- |
| `get_vectorstore()` | `@st.cache_resource` |
| `build_retriever()` | `@st.cache_resource` |
| `build_rag_chain()` | `@st.cache_resource` |
| `query_rag()`       | ❌ NO                 |

---

## 5️⃣ Ejemplo completo aplicado

```python
# retrievers.py
import streamlit as st

@st.cache_resource
def build_retriever() -> BaseRetriever:
    ...
```

```python
# rag.py
import streamlit as st

@st.cache_resource
def build_rag_chain():
    retriever = build_retriever()
    ...
    return rag_chain, retriever
```

---

## 6️⃣ Señal de que lo estás usando bien

✔ La app no se “reinicia” en cada pregunta
✔ Menos latencia tras la primera consulta
✔ Logs de inicialización solo una vez

---

## 🧠 Regla mental rápida

> **¿Este objeto es caro y no depende del input del usuario?**
> 👉 `cache_resource`

> **¿Depende del texto de la pregunta?**
> 👉 NO cachear

---

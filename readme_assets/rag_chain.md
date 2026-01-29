## 📌 Código completo del pipeline RAG actual

```python
rag_chain = (
    {
        "context": retriever | format_context,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm_generation
    | StrOutputParser()
)
```

Esto **NO es magia**, es **composición de Runnables (LCEL) en LangChain moderno**.

---

## 🧠 Idea clave antes de empezar

En tu implementación:

* Todo es un **Runnable**
* `|` significa:
  👉 *“la salida de la izquierda entra como input de la derecha”*

Piensa en esto como una **tubería de datos**: la pregunta entra, los documentos se recuperan, se formatean, se pasa al prompt y finalmente el LLM devuelve texto.

---

## 🧩 1️⃣ El bloque inicial (diccionario)

```python
{
    "context": retriever | format_context,
    "question": RunnablePassthrough(),
}
```

### ¿Qué hace?

👉 Es un **RunnableMap**: toma la entrada `query` y genera un diccionario:

```python
{
  "question": "consulta del usuario",
  "context": "texto de los documentos relevantes formateado"
}
```

---

### 🔹 `"question": RunnablePassthrough()`

* Recibe la pregunta del usuario.
* La pasa **tal cual** al prompt.
* Ejemplo:

```python
query = "¿Cómo puedo resetear mi contraseña?"
output["question"] = "¿Cómo puedo resetear mi contraseña?"
```

Sirve para que el prompt reciba la pregunta original.

---

### 🔹 `"context": retriever | format_context`

Aquí está la **magia RAG** 🔥

#### Paso 1: `retriever`

```python
docs = retriever.invoke(query)  # List[Document]
```

Devuelve objetos `Document` con contenido y metadata:

```python
[
  Document(page_content="Para resetear tu contraseña ...", metadata={"filename":"manual.pdf"}),
  Document(page_content="Sigue estos pasos ...", metadata={"filename":"faq.pdf"}),
]
```

#### Paso 2: `| format_context`

Convierte los `Document` en texto plano legible para el prompt, añade encabezados y fuentes:

```
[Document 1] - Source: manual.pdf
Para resetear tu contraseña ...

[Document 2] - Source: faq.pdf
Sigue estos pasos ...
```

Se asigna a `"context"` en el diccionario.

---

### ✅ Resultado del bloque inicial

Si la pregunta es:

```
"¿Cómo puedo resetear mi contraseña?"
```

El output será:

```python
{
  "question": "¿Cómo puedo resetear mi contraseña?",
  "context": "[Document 1]...\n\n[Document 2]..."
}
```

---

## 🧩 2️⃣ `| rag_prompt`

```python
| rag_prompt
```

El prompt RAG espera:

```
FRAGMENTOS DE SOPORTE:
{context}

PREGUNTA: {question}
```

LangChain reemplaza automáticamente `{context}` y `{question}` y genera un **string listo para el LLM**.

---

## 🧩 3️⃣ `| llm_generation`

```python
| llm_generation
```

* Envía el prompt al modelo (OpenAI, etc.)
* Devuelve la respuesta generada

Ejemplo conceptual:

```python
AIMessage(content="Para resetear tu contraseña, sigue estos pasos...")
```

---

## 🧩 4️⃣ `| StrOutputParser()`

* Extrae solo **texto plano**
* Elimina metadata o envoltorios del LLM

Resultado final:

```python
"Para resetear tu contraseña, sigue estos pasos..."
```

---

## 🧠 Diagrama mental completo

```
Pregunta del usuario
   │
   ▼
RunnableMap {
  question ──────────────► "¿Cómo puedo resetear mi contraseña?"
  context  ─► retriever ─► docs ─► format_context ─► texto
}
   │
   ▼
rag_prompt (inyecta context + question)
   │
   ▼
LLM (genera respuesta)
   │
   ▼
StrOutputParser
   │
   ▼
Respuesta final (str)
```

---

## 🧪 ¿Qué devuelve `query_rag(query)`?

```python
answer_obj: RagAnswer = query_rag("¿Cómo puedo resetear mi contraseña?")
```

`RagAnswer` incluye:

* `answer` → texto final para mostrar al usuario
* `confidence` → heurística de confiabilidad
* `sources` → lista de archivos que respaldan la respuesta

> Nota: **los documentos originales (`Document`) se recuperan por separado con `retriever.invoke(query)`** para mostrar fragmentos en la UI.

---

## 🧠 Por qué esta arquitectura es buena

✅ Separación clara:

* retrieval (recuperar documentos)
* formatting (contexto legible)
* prompting (prompt RAG)
* generation (LLM)

✅ Fácil de extender:

* reranking
* filtros
* explicaciones adicionales

✅ Transparente:

* `query_rag` da respuesta final
* `retriever.invoke` da trazabilidad de documentos

---

## 💡 Analogía

* `retriever` → biblioteca: devuelve los libros relevantes
* `format_context` → resumen legible de los libros
* `rag_chain` → abogado: lee los libros, genera respuesta
* `query_rag` → función que entrega **respuesta + confianza + fuentes** al usuario

---

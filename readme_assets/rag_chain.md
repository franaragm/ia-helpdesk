## 📌 El código completo

```python
rag_chain = (
    {
        "context": retriever | _format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm_generation
    | StrOutputParser()
)
```

Esto **NO es magia**, es **composición de Runnables** en LangChain.

---

## 🧠 Idea clave antes de empezar

En LangChain moderno:

* Todo es un **Runnable**
* `|` significa:
  👉 *“la salida de la izquierda entra como input de la derecha”*

Piensa en esto como una **tubería de datos**.

---

## 🧩 1️⃣ El bloque inicial (diccionario)

```python
{
    "context": retriever | _format_docs,
    "question": RunnablePassthrough(),
}
```

### ¿Qué es esto?

👉 Es un **RunnableMap**
Convierte **un solo input** (la pregunta) en **un diccionario estructurado**.

---

### 🔹 `"question": RunnablePassthrough()`

```python
"question": RunnablePassthrough()
```

* Recibe el input original (la pregunta)
* Lo devuelve **tal cual**
* Sirve para pasar la pregunta al prompt

Ejemplo:

```python
input = "¿Quién es el arrendatario?"
output["question"] = "¿Quién es el arrendatario?"
```

---

### 🔹 `"context": retriever | _format_docs`

Aquí está la **magia RAG** 🔥

#### Paso 1: `retriever`

```python
retriever.invoke(question) -> List[Document]
```

Devuelve algo así:

```python
[
  Document(page_content="El arrendatario es Juan Pérez...", metadata={...}),
  Document(page_content="Contrato firmado el 3 de mayo...", metadata={...})
]
```

---

#### Paso 2: `| _format_docs`

```python
retriever | _format_docs
```

* Toma la lista de `Document`
* Los convierte en **texto legible**
* Añade fuentes, páginas, numeración

Resultado final:

```text
[Fragmento 1] - Fuente: contrato1.pdf - Página: 2
El arrendatario es Juan Pérez...

[Fragmento 2] - Fuente: contrato2.pdf - Página: 1
Contrato firmado el 3 de mayo...
```

👉 Eso se asigna a la clave `"context"`.

---

### ✅ Resultado del bloque completo

Si la pregunta es:

```
"¿Quién es el arrendatario?"
```

El output de este bloque será:

```python
{
  "question": "¿Quién es el arrendatario?",
  "context": "[Fragmento 1]...\n\n[Fragmento 2]..."
}
```

---

## 🧩 2️⃣ `| rag_prompt`

```python
| rag_prompt
```

Tu prompt es:

```text
FRAGMENTOS DE CONTRATOS:
{context}

PREGUNTA: {question}
```

LangChain hace automáticamente:

```python
rag_prompt.format(
    context=context,
    question=question
)
```

👉 Resultado: **un string listo para el LLM**.

---

## 🧩 3️⃣ `| llm_generation`

```python
| llm_generation
```

* Envía el prompt al modelo (OpenAI / Groq / etc.)
* Devuelve la respuesta del LLM (objeto o mensaje)

Ejemplo conceptual:

```python
AIMessage(
  content="El arrendatario es Juan Pérez..."
)
```

---

## 🧩 4️⃣ `| StrOutputParser()`

```python
| StrOutputParser()
```

* Extrae solo el **texto plano**
* Elimina metadata del mensaje

Resultado final:

```python
"El arrendatario es Juan Pérez..."
```

---

## 🧠 Diagrama mental completo

```
Pregunta
   │
   ▼
RunnableMap {
  question ──────────────► "¿Quién es el arrendatario?"
  context  ─► retriever ─► docs ─► _format_docs ─► texto
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

## 🧪 ¿Qué devuelve `rag_chain.invoke()`?

```python
answer: str = rag_chain.invoke("¿Quién es el arrendatario?")
```

👉 **Solo la respuesta**, no los documentos
(por eso los recuperas aparte para la UI).

---

## 🧠 Por qué esta arquitectura es MUY buena

✅ Separación clara:

* retrieval
* formatting
* prompting
* generation

✅ Fácil de extender:

* añadir reranking
* añadir filtros
* añadir explicaciones

✅ 100% compatible con LangChain moderno

---



## **una diferencia conceptual entre “recuperar documentos” y “generar respuesta” en RAG**.

Vamos paso a paso:

---

### 1️⃣ Lo que hace `rag_chain.invoke(question)`

En tu pipeline:

```python
rag_chain = (
    {
        "context": retriever | _format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm_generation
    | StrOutputParser()
)
```

Cuando ejecutas:

```python
answer = rag_chain.invoke(question)
```

* **Internamente**, el pipeline hace:

  1. `retriever.invoke(question)` → obtiene los documentos
  2. `_format_docs` → los convierte en texto
  3. Inserta ese texto en `rag_prompt`
  4. Llama al LLM (`llm_generation`)
  5. Extrae el texto final (`StrOutputParser`)

✅ El resultado `answer` **ya incluye la información de los documentos**, pero **no tienes acceso a los objetos Document originales**.

---

### 2️⃣ Por qué necesitas invocar el retriever por separado

```python
docs = retriever.invoke(question)
```

* Esto te da **los objetos `Document` reales**.
* Incluyen metadata como:

  * `source` (archivo)
  * `page` (página)
  * `chunk_id`
* Que luego usas para mostrar los fragmentos en la UI o para **log/traceability**.

Si solo usaras `rag_chain.invoke()`, **solo tendrías texto plano**, sin saber de dónde vino cada fragmento.

---

### 3️⃣ Ejemplo conceptual

Pregunta:

```
"¿Quién es el arrendatario?"
```

3a) `rag_chain.invoke(question)` → `answer`

```
"El arrendatario es Juan Pérez..."
```

* Útil para mostrar al usuario
* No te dice **qué documento / página** respalda la respuesta

3b) `retriever.invoke(question)` → `docs`

```
[
  Document(page_content="El arrendatario es Juan Pérez", metadata={"source":"contrato1.pdf", "page":2}),
  Document(page_content="Contrato firmado...", metadata={"source":"contrato2.pdf", "page":1})
]
```

* Útil para mostrar **fragmentos**, referencias y auditoría
* Te permite construir UI “fragmento por fragmento” (lo que haces en tu columna derecha)

---

### 4️⃣ Por qué no se combinan directamente

Podrías intentar:

```python
answer, docs = rag_chain.invoke_and_return_docs(question)
```

Pero **LangChain no tiene un método estándar así**.
Separar **retrieval** y **generation** te da:

* Flexibilidad
* Mejor trazabilidad
* Posibilidad de **re-ranking** o post-procesamiento antes de la generación

---

### 5️⃣ Resumen conceptual

| Acción                                      | Método                       | Resultado                   | Uso en tu app                |
| ------------------------------------------- | ---------------------------- | --------------------------- | ---------------------------- |
| Recuperar documentos relevantes             | `retriever.invoke(question)` | List[Document] con metadata | Mostrar fragmentos en UI     |
| Generar respuesta basada en esos documentos | `rag_chain.invoke(question)` | str (texto de LLM)          | Mostrar respuesta al usuario |

---

💡 **Analogía:**

* `retriever` → biblioteca → te da los libros
* `rag_chain` → abogado → lee los libros y te responde
* Necesitas **los libros y la respuesta** para que todo sea transparente y auditable.

---

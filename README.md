# ⚖️ ASISTENTE LEGAL CON IA (RAG)

## 📝 Descripción

Asistente legal basado en **Retrieval-Augmented Generation (RAG)**, especializado en **contratos de arrendamiento**.

El sistema permite:

- 📥 Cargar y vectorizar contratos en PDF
- 📤 **Subir nuevos contratos desde la interfaz web**
- 🧠 **Indexación incremental** (los nuevos documentos se añaden sin borrar los anteriores)
- 🔎 Recuperar fragmentos relevantes usando búsqueda semántica avanzada
- 🤖 Generar respuestas **fundamentadas exclusivamente en los documentos**
- 📄 Mostrar los fragmentos utilizados como soporte de la respuesta

Está diseñado con una arquitectura modular y extensible, pensada para **casos legales reales**.

![screenshot](readme_assets/screenshot.png)

---

## 🐍 Requisitos de Python

* **Python 3.13.2** (recomendado, probado en macOS Apple Silicon y Windows)
* **Python 3.11** (ideal para Mac Intel)

⚠️ **No usar Python 3.14+**, ya que rompe compatibilidad con:

- Pydantic
- ChromaDB
- LangChain Core

---

## 📂 Estructura del proyecto

```

/ia-legal-assistant/
├── app/
│   ├── documents/              # PDFs iniciales (bootstrap opcional)
│   ├── loader.py               # Carga PDFs (directorio y uploads) y los divide en chunks
│   ├── rag.py                  # Orquestación del pipeline RAG
│   ├── retrievers.py           # Construcción de retrievers (MMR, MultiQuery, Hybrid)
│   ├── vectorstore.py          # Creación y carga del vectorstore Chroma (persistente)
│   ├── prompts.py              # Prompts del sistema (RAG, relevance)
│   ├── schemas.py              # Modelos Pydantic (RagResponse, RetrievedDocument)
│   ├── ui.py                   # Interfaz de usuario (Streamlit)
│   ├── bootstrap.py            # Inicialización segura de ChromaDB
│   └── services/
│       ├── llm_client.py       # Clientes LLM (OpenAI, Google, OpenRouter)
│       └── utils.py            # Utilidades (hash de texto, env vars, etc.)
├── run_app.py                  # Punto de entrada de la aplicación
├── config_base.py              # Configuración global (modelos, paths, RAG)
├── requirements.txt            # Dependencias principales
├── requirements.lock           # Dependencias fijadas
└── .env                        # Variables de entorno

```

---

## 🧠 Arquitectura RAG (resumen)

### 🔹 Inicialización (una sola vez)

```

PDFs iniciales
↓
load_documents
↓
create_vectorstore
↓
ChromaDB (persistente en disco)

```

> Solo se ejecuta si la colección está vacía.

---

### 🔹 Indexación incremental (desde la UI)

```

Usuario sube PDF
↓
Carga temporal del archivo
↓
Split en chunks
↓
Hash único por fragmento
↓
Inserción en ChromaDB

```

✔️ Los documentos existentes **no se borran**  
✔️ Se evitan duplicados mediante IDs hash  
✔️ El vectorstore se actualiza en caliente  

---

### 🔹 Flujo de consulta

```

Pregunta del usuario
↓
MultiQueryRetriever
↓
MMR Retriever
↓
(Opcional) Hybrid con Similarity
↓
Fragmentos relevantes
↓
LLM (rag_prompt)
↓
Respuesta + documentos citados


```

---

## 🚀 Instalación y uso

### 🔧 1) Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

---

### 📦 2) Instalar dependencias

Dos opciones:

```bash
pip install -r requirements.txt
pip install -r requirements.lock
```

#### Cuando se añade una nueva dependencia

```bash
pip install -r requirements.txt
pip freeze > requirements.lock
```

---

### 🔐 3) Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env`:

```env
OPENAI_API_KEY=API_KEY_HERE
GOOGLEAI_API_KEY=API_KEY_HERE
OPENROUTER_API_KEY=API_KEY_HERE
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
GROQ_API_KEY=API_KEY_HERE
GROQ_BASE_URL=https://api.groq.com/openai/v1
ENV=dev
```

#### 🔑 Obtener API keys

* OpenAI → [https://platform.openai.com/settings/organization/api-keys](https://platform.openai.com/settings/organization/api-keys)
* OpenRouter → [https://openrouter.ai/keys](https://openrouter.ai/keys)
* Google AI → [https://aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)
* Groq → [https://console.groq.com/keys](https://console.groq.com/keys)

---

### ▶️ 4) Ejecutar la aplicación

```bash
streamlit run run_app.py
```

Disponible en:

```
http://localhost:8501
```

---

## 🖥️ Uso de la interfaz

### 💬 Chat legal

* Escribe una consulta sobre contratos
* El asistente responde **solo usando los documentos indexados**
* No inventa información fuera del contexto

### 📄 Documentos relevantes

* En la columna derecha se muestran:

  * Fragmento utilizado
  * Archivo de origen
  * Página del documento

### 📤 Subir nuevos contratos (NEW)

Desde la **barra lateral**:

* Sube un PDF de contrato
* El documento se indexa automáticamente
* Pasa a estar disponible para futuras consultas
* No es necesario reiniciar la aplicación

---

## 🧪 Funcionalidades experimentales (otras ramas)

Este repositorio incluye **ramas experimentales** con funcionalidades avanzadas que **no están activas en `main`**, entre ellas:

* 🧠 **Filtrado por relevancia con LLM**
  - Evaluación semántica de fragmentos antes de la generación
  - Activación automática según tipo de pregunta
  - Control estricto de coste (modelo barato + límites)

Estas features se mantienen separadas para:
- Preservar estabilidad
- Evitar sobrecostes innecesarios
- Facilitar experimentación controlada

---

## 🛠️ Desarrollo y extensibilidad

El proyecto está preparado para añadir fácilmente:

* 🔎 Re-ranking legal avanzado
* 🧾 Extracción de entidades (personas, importes, fechas)
* 📜 Versionado de contratos
* 📊 Evaluación del RAG (precision / recall)
* 🌐 API REST con FastAPI
* 🧠 Agentes legales / LangGraph

---

## 📌 Notas importantes

* ChromaDB es **persistente** (no se pierde información al reiniciar)
* La indexación es **incremental y segura**
* La subida de PDFs usa archivos temporales
* No se reindexan documentos duplicados
* Diseñado para minimizar alucinaciones en contexto legal
* El filtrado por relevancia con LLM se desarrolla en una rama separada


---

## 📚 Recursos

* LangChain → [https://www.langchain.com/](https://www.langchain.com/)
* Streamlit → [https://streamlit.io/](https://streamlit.io/)
* ChromaDB → [https://www.trychroma.com/](https://www.trychroma.com/)
* Pydantic → [https://docs.pydantic.dev/](https://docs.pydantic.dev/)
* PyPDF → [https://pypdf.readthedocs.io/](https://pypdf.readthedocs.io/)





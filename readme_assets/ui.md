# 🧠 Dos escenarios distintos en el flujo del Helpdesk con LangGraph

El sistema de helpdesk con RAG y LangGraph tiene dos momentos claramente diferenciados en el procesamiento de un ticket:

---

## 1️⃣ Ejecución inicial del ticket — `process_query(...)`

**Cuándo se usa:**
*Se llama **una sola vez** cuando un usuario crea un ticket o envía una consulta por primera vez.*

### Qué hace:

1. **Crea el estado inicial del ticket**

   * Se instancia un objeto `HelpdeskState` con la consulta del usuario, historial vacío y valores iniciales.
   * Se prepara un `ticket_id` como `thread_id` para checkpointing.

2. **Configura checkpointing y streaming**

   * `config` con `thread_id` permite pausar y reanudar el grafo más adelante si fuera necesario.
   * `processing_history` inicializa la lista que almacenará todos los logs explicativos de cada nodo.

3. **Ejecuta el grafo desde START**

   * `st.session_state.helpdesk.stream(...)` recorre todos los nodos del grafo.
   * Cada nodo puede devolver **salidas parciales** (streaming) y **historial** que se acumula en `processing_history`.
   * Esto permite ver progresos en tiempo real si se conecta a la UI.

4. **Obtiene el estado final consolidado**

   * `get_state(config)` devuelve el estado completo del grafo, incluyendo campos de `HelpdeskState` y metadata interna.
   * Se valida con `HelpdeskStateModel(...).model_dump(exclude_none=True)` para garantizar consistencia y eliminar campos vacíos.

5. **Devuelve la información a la UI**

   * `final_state`: dict con los valores finales del ticket
   * `processing_history`: historial completo de pasos ejecutados
   * `config`: configuración usada, útil para reanudar o actualizar el estado

### Resumen conceptual:

| Acción       | Método                                          | Resultado                                     |
| ------------ | ----------------------------------------------- | --------------------------------------------- |
| Crear ticket | `process_query(...)`                            | Estado inicial + ejecución completa del grafo |
| Streaming    | Parcial mientras se ejecuta                     | Logs e historial                              |
| Salida final | `final_state` + `processing_history` + `config` | Se muestra en UI y se guarda                  |

✅ **Responsabilidad:** ejecución inicial completa de un ticket.
✅ **Uso:** una vez por ticket.

---

## 2️⃣ Reanudación tras intervención humana — `resume_with_human_answer(...)`

**Cuándo se usa:**
*Se llama **0 o N veces** solo si el ticket fue escalado a un humano y se necesita continuar el flujo.*

### Qué hace:

1. **Inyecta la respuesta del agente humano en el estado existente**

   * `update_state(ticket_config, {"human_answer": human_answer})`
   * El flujo del grafo ya tiene checkpoints; solo actualizamos el valor que faltaba.

2. **Reanuda el grafo desde el punto de interrupción**

   * `stream(None, config=ticket_config, stream_mode="updates")`
   * Pasamos `None` porque el estado inicial ya existe en el checkpoint.
   * Solo se ejecutan los nodos pendientes (por ejemplo, `process_human` y `final_answer`).
   * Cada nodo puede devolver historial parcial, que se acumula en `resumed_history`.

3. **Obtiene el estado final consolidado**

   * Igual que en `process_query`, se usa `get_state(ticket_config)` y se valida con `HelpdeskStateModel(...)`.
   * Devuelve un estado limpio listo para la UI y persistencia.

### Resumen conceptual:

| Acción                    | Método                                | Resultado                                        |
| ------------------------- | ------------------------------------- | ------------------------------------------------ |
| Inyectar respuesta humana | `resume_with_human_answer(...)`       | El flujo continua desde el checkpoint            |
| Streaming                 | Solo nodos pendientes                 | Historial de pasos posteriores a la intervención |
| Salida final              | `validated_state` + `resumed_history` | Estado actualizado del ticket y logs             |

✅ **Responsabilidad:** continuar un flujo pausado tras intervención humana.
✅ **Uso:** solo si el ticket requiere escalado.

---

## ⚡ Diferencias clave entre los métodos

| Aspecto            | `process_query`                  | `resume_with_human_answer`              |
| ------------------ | -------------------------------- | --------------------------------------- |
| Estado inicial     | Sí, se crea desde cero           | No, se usa checkpoint existente         |
| Nodo de inicio     | START                            | Nodo interrumpido (ej. `process_human`) |
| Ejecución completa | Sí                               | Solo nodos pendientes                   |
| Historial          | Se acumula desde el inicio       | Se acumula desde el último checkpoint   |
| Uso por ticket     | 1 vez                            | 0 o N veces, según intervención humana  |
| Checkpointing      | Se establece para la primera vez | Se reutiliza para continuar el flujo    |

---

### ✅ Conclusión conceptual

* **`process_query(...)`** → ejecución inicial, prepara todo el ticket y el grafo desde cero.
* **`resume_with_human_answer(...)`** → reanudación incremental, solo se inyecta la información humana y se continúa el flujo.
* Son **responsabilidades distintas**, por eso se implementan como métodos separados.
* Mantenerlos separados asegura claridad, robustez y facilidad de mantenimiento.

---

Aquí tienes un desglose de los **puntos importantes y responsabilidades clave** del método `main()` de tu UI Helpdesk con RAG + ChromaDB, explicado de manera estructurada y clara:

---

# 🧠 Explicación de la función `main()`

`main()` es la **UI principal del sistema**, construida con **Streamlit**, que combina interacción de usuario, control de tickets y flujo RAG + intervención humana.

Se puede dividir en **secciones lógicas**:

---

## 1️⃣ Cabecera y descripción

```python
st.title("🎧 Helpdesk 2.0 con RAG + ChromaDB")
st.markdown("*Sistema inteligente con LangGraph y búsqueda vectorial*")
```

* Muestra **título y descripción** de la aplicación.
* Contextualiza al usuario sobre la combinación de **RAG + LangGraph + vectorstore**.

---

## 2️⃣ Verificación del sistema RAG

```python
is_rag_configured = check_rag_setup()
```

* Comprueba si **ChromaDB y RAG están inicializados**.
* Si no, se bloquea la funcionalidad principal y se pide configurar desde la barra lateral.

---

## 3️⃣ Barra lateral (sidebar)

Se encarga de **control y configuración del sistema**:

### Panel de control

```python
st.metric("Tickets Activos", len(st.session_state.tickets))
```

* Muestra el número de tickets actualmente activos en sesión.

### Estado RAG

* Muestra **si ChromaDB está configurado**.
* Permite **configurar o reconfigurar RAG** con botones.
* Mensajes visuales (`success`, `warning`, `error`) indican el estado.

### Flujo del sistema

* Explica **el pipeline de procesamiento** de un ticket:

  1. Usuario envía consulta
  2. Clasificación automática
  3. Búsqueda vectorial RAG
  4. Evaluación de confianza
  5. Escalado humano si es necesario
  6. Respuesta final

### Configuración extra

* Botón para **limpiar todos los tickets** de la sesión.

> ✅ La sidebar es **administrativa**: controla el estado y permite intervenir sin entrar en los detalles de cada ticket.

---

## 4️⃣ Área principal dividida en columnas

```python
col1, col2 = st.columns([1, 1])
```

* `col1`: Nueva consulta del usuario
* `col2`: Tickets recientes

---

### 4a️ Nueva consulta (col1)

* Formulario para **enviar un ticket nuevo**:

1. **Selector de ejemplos** (`selectbox`)

   * Permite elegir consultas predefinidas (`HELPDESK_EXAMPLES`).

2. **Campos de usuario y descripción** (`text_input` y `text_area`)

3. **Envío del ticket** (`form_submit_button`)

* Genera un `ticket_id` único con `generate_uuid()`.
* Ejecuta **process_query(query, ticket_id)** para:

  * Crear el estado inicial
  * Ejecutar el grafo LangGraph
  * Generar RAG + historial
* Guarda en `st.session_state.tickets` toda la información:

  * Usuario, consulta, resultado RAG, historial, config, timestamp

> ✅ Aquí ocurre la **creación inicial de un ticket** y el disparo del pipeline de RAG.

---

### 4b️ Tickets recientes (col2)

* Lista los tickets más recientes en orden inverso (los más nuevos primero).

* Para cada ticket:

  * Muestra **usuario, consulta y timestamp**
  * Muestra **historial de procesamiento** (`history`)
  * Muestra **categoría** si está definida (`category`)
  * Información RAG:

    * Confianza (`confidence`) con barra de progreso
    * Fuentes consultadas (`sources`)

* **Human-in-the-loop**:

  * Si `requires_human` es True y no hay `final_answer`:

    * Se permite **escribir respuesta humana**
    * Botones:

      * **Enviar respuesta humana** → llama a `resume_with_human_answer()`
      * **Usar respuesta RAG** → reusa RAG como “respuesta humana”

* **Respuesta final**:

  * Si `final_answer` existe, se muestra con métricas:

    * Confianza, número de fuentes, quién resolvió (humano o RAG)

> ✅ Esta columna maneja **interacción posterior a la creación del ticket**, incluyendo visualización, intervención humana y métricas finales.

---

## 5️⃣ Resumen de responsabilidades clave de `main()`

| Sección            | Función principal                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Cabecera           | Mostrar título y descripción de la app                                                                                   |
| Sidebar            | Panel de control, métricas de tickets, estado/configuración RAG, limpieza de tickets                                     |
| Nueva consulta     | Crear ticket, ejecutar `process_query()`, enviar al pipeline RAG, almacenar resultado                                    |
| Tickets recientes  | Mostrar tickets, historial, categoría, confianza, fuentes, respuesta final, intervención humana                          |
| Human-in-the-loop  | Permitir que un agente humano o el sistema RAG complete la respuesta, actualizar estado con `resume_with_human_answer()` |
| Validación y rerun | Asegurar que la UI se actualice dinámicamente tras creación o actualización de tickets                                   |

---

### ⚡ Puntos importantes a destacar

1. **Integración completa de RAG + LangGraph**

   * El pipeline de RAG se ejecuta al crear un ticket.
   * El historial y estado se mantiene para trazabilidad.

2. **Separación UI / Lógica**

   * `main()` solo maneja visualización e interacción.
   * Toda la lógica de RAG, retrievers y grafo está en módulos externos (`process_query`, `resume_with_human_answer`, `build_retriever`).

3. **Human-in-the-loop**

   * Permite reanudar el flujo del ticket tras intervención humana sin reiniciar todo el pipeline.

4. **Persistencia en sesión**

   * Todos los tickets se guardan en `st.session_state.tickets`.
   * Permite ver tickets recientes y mantener estado entre interacciones.

5. **Manejo dinámico del sistema RAG**

   * La aplicación comprueba configuración antes de permitir consultas.
   * Se puede reconfigurar en caliente sin reiniciar la app.

6. **Experiencia de usuario**

   * Streaming, barras de progreso, historial detallado, métricas y feedback visual para cada ticket.

---

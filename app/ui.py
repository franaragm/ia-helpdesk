import streamlit as st
from datetime import datetime
from .services.utils import generate_uuid
from .constants import HELPDESK_EXAMPLES
from .graph import compile_helpdesk
from .schemas import HelpdeskState, HelpdeskStateModel
from .bootstrap import init_chroma
from .vectorstore import get_vectorstore


# ======================================================
# Configuración de la página
# ======================================================

st.set_page_config(
    page_title="Helpdesk 2.0 con RAG",
    page_icon="🎧",
    layout="wide"
)

# ======================================================
# Inicialización del estado de sesión
# ======================================================
# Se crea UNA única instancia del grafo con checkpointing
# y se mantiene viva durante toda la sesión del usuario
if "helpdesk" not in st.session_state:
    st.session_state.helpdesk = compile_helpdesk()

if "tickets" not in st.session_state:
    st.session_state.tickets = {}
    
if "example_query" not in st.session_state:
    st.session_state.example_query = ""

# ======================================================
# Utilidades de configuración RAG
# ======================================================

def check_rag_setup() -> bool:
    """
    Verifica si el sistema RAG está correctamente configurado.
    
    - Intenta cargar el vectorstore persistido.
    - Comprueba que tenga documentos indexados.
    - Retorna True si el vectorstore funciona y contiene al menos un chunk.
    """
    try:
        with st.spinner("🔍 Verificando configuración RAG..."):
            # Cargar el vectorstore cacheado
            vectorstore = get_vectorstore()
            
            # Contar documentos/chunks indexados
            doc_count = len(vectorstore.get()["ids"])
            
            if doc_count == 0:
                st.warning("⚠️ Vectorstore OK pero no hay documentos indexados.")
                return False
            
            st.success(f"✅ Vectorstore cargado correctamente. Documentos indexados: {doc_count}")
            return True
    
    except Exception as e:
        st.error(f"❌ Error verificando RAG: {e}")
        return False


def configure_rag() -> bool:
    """
    Inicializa o actualiza ChromaDB de forma incremental.

    - Muestra un spinner mientras se ejecuta la indexación
    - Devuelve True si el proceso finaliza correctamente
    - Devuelve False si ocurre algún error
    """
    try:
        with st.spinner("🔧 Configurando sistema RAG..."):
            init_chroma()
        return True
    except Exception as e:
        st.error(f"❌ Error configurando RAG: {e}")
        return False


# ======================================================
# Ejecución del grafo LangGraph
# ======================================================

def process_query(query: str, ticket_id: str) -> tuple[dict, list[str], dict]:
    """
    Ejecuta el grafo LangGraph usando streaming y checkpointing para procesar una consulta.

    Flujo del método:
    1. Se crea un estado inicial del helpdesk con la consulta del usuario.
    2. Se define un `thread_id` en la configuración para poder reanudar ejecuciones pausadas.
    3. Se itera sobre los eventos parciales del grafo (streaming).
       - Cada evento puede contener la salida de varios nodos.
       - Se acumula el historial explicativo de cada nodo.
    4. Se obtiene el estado final consolidado del grafo.
    5. Se devuelve:
       - `final_state.values`: el estado final en formato dict, listo para la UI.
       - `processing_history`: lista completa de pasos o logs del procesamiento.
       - `config`: configuración usada, útil para reanudar o actualizar el estado.

    Parámetros:
    - query (str): Texto de la consulta del usuario.
    - ticket_id (str): Identificador único del ticket, usado como thread_id.

    Retorna:
    - Tuple[dict, List[str], dict]: (estado final, historial completo, configuración)
    """

    # ================================
    # 1. Crear el estado inicial
    # ================================
    initial_state = HelpdeskState(
        query=query,                # Consulta original
        category=None,              # Categoría del ticket (automática o escalada)
        rag_answer=None,            # Respuesta generada por RAG
        confidence=0.0,             # Confianza heurística
        sources=[],                 # Fuentes consultadas
        rag_context=None,           # Contexto textual usado por RAG
        requires_human=False,       # Flag si necesita intervención humana
        human_answer=None,          # Respuesta del humano si aplica
        final_answer=None,          # Respuesta final consolidada
        history=[]                  # Historial explicativo del flujo
    )

    # ================================
    # 2. Configuración para checkpointing
    # ================================
    # `thread_id` permite pausar y reanudar la ejecución del grafo
    config = {"configurable": {"thread_id": ticket_id}}

    # Lista donde vamos a acumular todo el historial de pasos del grafo
    processing_history: list[str] = []

    try:
        # ================================
        # 3. Streaming de actualizaciones del grafo
        # ================================
        # El método stream() devuelve un iterable de stream_event
        # Cada stream_event es un dict {nodo: salida_parcial}
        # Esto permite mostrar progresos en tiempo real si quisiéramos
        for stream_event in st.session_state.helpdesk.stream(
            initial_state,
            config=config,
            stream_mode="updates"  # streaming parcial, recibe eventos a medida que se generan
        ):
            # Cada evento puede contener la salida de varios nodos
            for node, node_output in stream_event.items():
                # Si el nodo devuelve historial, se acumula en processing_history
                if "history" in node_output and node_output["history"]:
                    processing_history.extend(node_output["history"])

         # ================================
        # 4. Obtener estado final consolidado
        # ================================
        # El objeto final_state contiene TODO el estado interno del grafo:
        # - Valores de HelpdeskState
        # - Metadata interna
        # - Checkpoints, referencias a nodos, etc.
        final_state = st.session_state.helpdesk.get_state(config)
        
        # ================================
        # 5. Validar con Pydantic
        # ================================
        # final_state.values -> Diccionario limpio con solo los campos de HelpdeskState
        # Esto es seguro para guardar en st.session_state.tickets y mostrar en la UI
        validated_final_state = HelpdeskStateModel(**final_state.values).model_dump(exclude_none=True)


        # ================================
        # 6. Retornar resultados
        # ================================
        return validated_final_state, processing_history, config

    except Exception as e:
        # Cualquier error se muestra en la UI y retornamos valores vacíos
        st.error(f"❌ Error procesando consulta: {str(e)}")
        return None, [], None

# ======================================================
# Reanudación grafo tras intervención humana
# ======================================================

def resume_with_human_answer(ticket_config: dict, human_answer: str,) -> tuple[dict, list[str]]:
    """
    Reanuda la ejecución del grafo LangGraph tras una intervención humana.

    Flujo:
    1. Inyecta la respuesta del agente humano en el estado existente.
    2. Reanuda el grafo desde el último checkpoint (antes de process_human).
    3. Consume los eventos de streaming restantes.
    4. Devuelve el estado final validado y el historial generado.
    """

    resumed_history: list[str] = []

    try:
        # ================================
        # 1️⃣ Actualizar el estado del grafo con la respuesta humana
        # ================================
        # Inyectamos la respuesta escrita por el agente en el estado del grafo
        # `ticket_config` permite identificar el thread/ticket correcto
        # Esto es lo que hace que el flujo pueda continuar desde donde se pausó
        st.session_state.helpdesk.update_state(
            ticket_config,
            {"human_answer": human_answer},
        )

        # ================================
        # 2️⃣ Reanudar el procesamiento del grafo desde el punto de interrupción
        # ================================
        # El método stream() permite ejecutar nodos pendientes
        # pasando None como estado inicial porque ya tenemos el checkpoint
        # stream_mode="updates" devuelve eventos parciales (historial) mientras se ejecuta
        for stream_event in st.session_state.helpdesk.stream(
            None,  # None indica "continúa desde el último estado guardado"
            config=ticket_config,
            stream_mode="updates",
        ):  
            # Cada event puede contener la salida de varios nodos
            for _, node_output in stream_event.items():
                # Si el nodo devuelve historial de pasos, lo añadimos al ticket
                if node_output.get("history"):
                    resumed_history.extend(node_output["history"])

        # ================================
        # 3️⃣ Obtener el estado final consolidado del grafo
        # ================================
        # final_state contiene todo el estado interno
        # final_state.values -> solo los campos definidos en HelpdeskState
        final_state = st.session_state.helpdesk.get_state(ticket_config)

        # ================================
        # 4️⃣ Validar y guardar el estado final
        # ================================
        # Se valida con Pydantic para asegurar consistencia
        # model_dump(exclude_none=True) elimina campos vacíos para mantener el estado limpio
        validated_state = (
            HelpdeskStateModel(**final_state.values)
            .model_dump(exclude_none=True)
        )

        return validated_state, resumed_history

    except Exception as e:
        st.error(f"❌ Error reanudando ejecución con respuesta humana: {e}")
        return {}, []


# ======================================================
# Aplicación principal
# ======================================================

def main():
    """UI principal del sistema Helpdesk."""

    st.title("🎧 Helpdesk 2.0 con RAG + ChromaDB")
    st.markdown("*Sistema inteligente con LangGraph y búsqueda vectorial*")

    # Verificar estado del sistema RAG
    is_rag_configured = check_rag_setup()

    # ==================================================
    # Sidebar
    # ==================================================
    with st.sidebar:
        st.header("📊 Panel de Control")
        st.metric("Tickets Activos", len(st.session_state.tickets))

        # Estado del sistema RAG
        st.subheader("🔍 Estado RAG")
        if is_rag_configured:
            st.success("✅ ChromaDB configurado")
            if st.button("🔄 Reconfigurar RAG"):
                if configure_rag():
                    st.success("✅ RAG reconfigurado")
                    st.rerun()
                else:
                    st.error("❌ Error reconfigurando RAG")
        else:
            st.warning("⚠️ RAG no configurado")
            if st.button("🚀 Configurar RAG"):
                if configure_rag():
                    st.success("✅ RAG configurado exitosamente")
                    st.rerun()
                else:
                    st.error("❌ Error configurando RAG")

        st.subheader("🔄 Flujo del Sistema")
        st.text(
            """
            1. 📝 Usuario envía consulta
            2. 🤖 Clasificación automática
            3. 🔍 Búsqueda vectorial RAG
            4. 📊 Evaluación de confianza
            5. 👨‍💼 Escalado si es necesario
            6. ✅ Respuesta final
            """
        )

        st.subheader("⚙️ Configuración")
        if st.button("🗑️ Limpiar Tickets"):
            st.session_state.tickets = {}
            st.rerun()

    if not is_rag_configured:
        st.warning(
            "⚠️ El sistema RAG no está configurado. "
            "Usa el botón en la barra lateral para configurarlo."
        )
        return

    # ==================================================
    # Área principal
    # ==================================================
    col1, col2 = st.columns([1, 1])

    # ==================================================
    # Nueva consulta
    # ==================================================
    with col1:
        st.subheader("📝 Nueva Consulta")

        # Selectbox para elegir ejemplo
        selected_example = st.selectbox(
            "💡 Elige un ejemplo de consulta o deja vacío para escribir la tuya",
            options=[""] + HELPDESK_EXAMPLES,
            index=0
        )

        # Guardar selección en sesión
        if selected_example:
            st.session_state.example_query = selected_example

        with st.form("new_query"):
            user = st.text_input("👤 Usuario", placeholder="tu@email.com")

            initial_query = st.session_state.get("example_query", "")
            query = st.text_area(
                "💬 Descripción del problema",
                value=initial_query,
                placeholder="Describe tu consulta o problema aquí...",
                height=100
            )

            submitted = st.form_submit_button("🚀 Enviar Consulta")

            if submitted and query.strip():
                if "example_query" in st.session_state:
                    del st.session_state.example_query

                ticket_id = generate_uuid()

                with st.spinner("🔄 Procesando consulta..."):
                    result, history, config = process_query(query, ticket_id)

                if result:
                    st.session_state.tickets[ticket_id] = {
                        "user": user,
                        "query": query,
                        "result": result,
                        "history": history,
                        "config": config,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }

                    st.success(f"✅ Ticket {ticket_id} creado")
                    st.rerun()
                    
    # ==================================================
    # Tickets recientes
    # ==================================================
    with col2:
        st.subheader("🎫 Tickets Recientes")

        if not st.session_state.tickets:
            st.info("No hay tickets activos")
        else:
            # Mostrar tickets más recientes primero
            # `st.session_state.tickets` es un dict que mantiene el orden de inserción.
            # `.items()` devuelve pares (ticket_id, ticket_data) en orden de creación.
            # `list(...)` permite invertir el orden, ya que los iteradores no son reversibles.
            # `reversed(...)` hace que los tickets más recientes se muestren primero en la UI.
            for ticket_id, ticket_data in reversed(list(st.session_state.tickets.items())):
                with st.expander(f"🎫 {ticket_id} - {ticket_data['timestamp']}", expanded=False):

                    st.markdown(f"**👤 Usuario:** {ticket_data.get('user', '—')}")
                    st.markdown(f"**💬 Consulta:** {ticket_data['query']}")

                    result = ticket_data["result"]
                    history = ticket_data["history"]

                    # ----------------------------
                    # Historial de procesamiento
                    # ----------------------------
                    if history:
                        st.subheader("🔄 Procesamiento")
                        for step in history:
                            st.text(step)

                    # ----------------------------
                    # Clasificación
                    # ----------------------------
                    if result.get("category"):
                        st.markdown(f"**📂 Categoría:** {result['category']}")

                    # ----------------------------
                    # Información RAG
                    # ----------------------------
                    confidence = result.get("confidence", 0.0)
                    if confidence > 0:
                        st.markdown(f"**🎯 Confianza RAG:** {confidence:.2f}")
                        st.progress(confidence)

                        if result.get("sources"):
                            st.markdown(
                                f"**📚 Fuentes:** {', '.join(result['sources'])}"
                            )

                    # ----------------------------
                    # Human-in-the-loop: intervención humana
                    # ----------------------------
                    if result.get("requires_human") and not result.get("final_answer"):
                        st.warning("👨‍💼 Requiere intervención humana")

                        if result.get("rag_answer"):
                            with st.expander("📋 Contexto RAG"):
                                st.text(result["rag_answer"])

                        human_reply = st.text_area(
                            "✍️ Respuesta del agente",
                            key=f"human_{ticket_id}",
                            height=100,
                            placeholder="Escribe la respuesta para el usuario..."
                        )

                        col_btn1, col_btn2 = st.columns(2)

                        with col_btn1:
                            if st.button("💾 Enviar Respuesta", key=f"send_{ticket_id}"):
                                if human_reply.strip():
                                    config = ticket_data["config"]
                                    
                                    # Actualizar el estado con la respuesta humana
                                    result, history = resume_with_human_answer(config, human_reply)
                                    ticket_data["result"] = result
                                    ticket_data["history"].extend(history)

                                    st.success("✅ Respuesta enviada")
                                    st.rerun()
                                else:
                                    st.warning("⚠️ Escribe una respuesta antes de enviar")

                        with col_btn2:
                            if st.button("🤖 Usar RAG", key=f"rag_{ticket_id}"):
                                config = ticket_data["config"]
                                
                                # Reutilizamos el mismo flujo que para el humano,
                                # pero pasando la respuesta RAG como si fuera la "humana"
                                result, history = resume_with_human_answer(
                                    ticket_config=config,
                                    human_answer=result.get("rag_answer", "")
                                )
                                ticket_data["result"] = result
                                ticket_data["history"].extend(history)

                                st.success("✅ Respuesta RAG aplicada")
                                st.rerun()

                    # ----------------------------
                    # Respuesta final
                    # ----------------------------
                    elif result.get("final_answer"):
                        st.success("✅ Ticket Resuelto")
                        st.markdown("**💬 Respuesta final:**")
                        
                        # Mostrar respuesta final
                        st.info(result["final_answer"])

                        # Métricas finales
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("🎯 Confianza", f"{confidence:.2f}")
                        with col_m2:
                            st.metric("📚 Fuentes", len(result.get("sources", [])))
                        with col_m3:
                            resolver = "Humano" if result.get("requires_human") else "RAG"
                            st.metric("🤖 Resuelto por", resolver)

    # ==================================================
    # Footer con estadísticas
    # ==================================================
    st.markdown("---")

    if st.session_state.tickets:
        total = len(st.session_state.tickets)

        resolved_rag = sum(
            1 for t in st.session_state.tickets.values()
            if t["result"].get("final_answer") and not t["result"].get("requires_human")
        )

        resolved_human = sum(
            1 for t in st.session_state.tickets.values()
            if t["result"].get("final_answer") and t["result"].get("requires_human")
        )

        pending = total - resolved_rag - resolved_human

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("📊 Total Tickets", total)
        with col_s2:
            st.metric("🤖 Resueltos por RAG", resolved_rag)
        with col_s3:
            st.metric("👨‍💼 Resueltos por Humano", resolved_human)
        with col_s4:
            st.metric("⏳ Pendientes", pending)

    st.markdown(
        """
        <div style='text-align: center'>
            <small>
                🚀 Powered by LangGraph · 🔍 ChromaDB · 🔄 Streaming · 💾 Checkpointing · 👨‍💼 Human-in-the-Loop
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )

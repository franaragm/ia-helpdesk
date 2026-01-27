import streamlit as st
from .rag import query_rag, get_retriever_info
from .loader import index_uploaded_pdf

def main():
    # Configuración de la página
    st.set_page_config(
        page_title="Asistente Legal RAG",
        page_icon="⚖️",
        layout="wide",
    )

    # Título
    st.title("⚖️ Asistente Legal basado en RAG")
    st.divider()

    # Inicializar el estado de la sesión
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Inicializar lista de archivos ya indexados
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

    # -------------------------------
    # Sidebar: carga de PDFs + info
    # -------------------------------
    with st.sidebar:
        st.header("📋 Información del Sistema")
        
        # ---------------------------
        # Subida de nuevo PDF
        # ---------------------------
        st.markdown("### 📤 Añadir documento PDF")
        uploaded_file = st.file_uploader(
            "Sube un contrato en PDF",
            type=["pdf"],
            accept_multiple_files=False,
        )
        
        if uploaded_file is not None:
            # Solo indexar si no se había subido antes
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner(f"🧠 Indexando '{uploaded_file.name}'..."):
                    chunks = index_uploaded_pdf(uploaded_file)
                    st.session_state.uploaded_files.add(uploaded_file.name)

                st.success(f"✅ Documento indexado ({chunks} fragmentos)")
                # No es necesario rerun aquí; Streamlit refresca automáticamente el estado

        # ---------------------------
        # Información del retriever
        # ---------------------------
        st.markdown("**🔍 Retriever**")
        retriever_info = get_retriever_info()
        st.info(f"Tipo: {retriever_info['tipo']}")
        st.info(f"Documentos: {retriever_info['documentos']}")
        st.info(f"Umbral: {retriever_info['umbral']}")
        
        st.markdown("**🤖 Modelos:**")
        st.info("Consultas: GPT-4o-mini\nRespuestas: GPT-4o")

        st.divider()

        # Botón para limpiar chat
        if st.button("🗑️ Limpiar chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # -------------------------------
    # Layout principal: Chat + Docs
    # -------------------------------    
    col1, col2 = st.columns([2, 1])

    # Chat
    with col1:
        st.markdown("### 💬 Chat")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Documentos relevantes
    with col2:
        st.markdown("### 📄 Documentos relevantes")

        if st.session_state.messages:
            last = st.session_state.messages[-1]
            if last["role"] == "assistant" and "docs" in last:
                for doc in last["docs"]:
                    with st.expander(f"📄 Fragmento {doc.fragmento}"):
                        st.markdown(f"**Fuente:** {doc.fuente}")
                        st.markdown(f"**Página:** {doc.pagina}")
                        st.text(doc.contenido)

    # -------------------------------
    # Input del usuario
    # -------------------------------
    if question := st.chat_input("Escribe tu consulta sobre contratos de arrendamiento..."):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.spinner("🔍 Analizando documentos..."):
            rag_response = query_rag(question)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": rag_response.answer,
                "docs": rag_response.documents,
            }
        )

        st.rerun()

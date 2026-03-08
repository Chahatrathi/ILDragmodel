import streamlit as st
import os
import shutil
from ingestion import main as run_ingestion
from historybased import ask_question

st.set_page_config(page_title="ILD Specialist Bot", page_icon="🫁")
st.title("🫁 ILD RAG Expert")

# --- SIDEBAR FOR FILE MANAGEMENT ---
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload ILD PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            # Save files to ./documents
            if not os.path.exists("./documents"):
                os.makedirs("./documents")
            
            for uploaded_file in uploaded_files:
                with open(os.path.join("./documents", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Run Ingestion
            with st.spinner("Indexing documents..."):
                run_ingestion()
                st.success("Database ready!")
        else:
            st.warning("Please upload files first.")

# --- CHAT INTERFACE ---
if not os.path.exists("./chroma_db"):
    st.info("👋 Welcome! Please upload your ILD research PDFs in the sidebar to begin.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about ILD..."):
    st.chat_message("user").markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            answer = ask_question(prompt)
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})

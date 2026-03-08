import streamlit as st
import os
from ingestion import main as run_ingestion
from historybased import ask_question

st.set_page_config(page_title="ILD Specialist Bot", page_icon="🫁")
st.title("🫁 ILD RAG Expert")

with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Build Knowledge Base"):
        if uploaded_files:
            if not os.path.exists("./documents"): os.makedirs("./documents")
            for f in uploaded_files:
                with open(os.path.join("./documents", f.name), "wb") as buffer:
                    buffer.write(f.getbuffer())
            with st.spinner("Indexing your files..."):
                run_ingestion()
        else:
            st.warning("Upload files first!")

if not os.path.exists("./chroma_db"):
    st.info("👈 Please upload PDFs in the sidebar and click 'Build Knowledge Base'.")
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
            try:
                answer = ask_question(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")

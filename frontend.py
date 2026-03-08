import streamlit as st
import os
from ingestion import main as run_ingestion
from historybased import ask_question

st.set_page_config(page_title="ILD Specialist Bot", page_icon="🫁")
st.title("🫁 ILD RAG Expert")

# Startup: Check if data is processed
@st.cache_resource
def check_db():
    if not os.path.exists("./chroma_db"):
        with st.spinner("Processing documents for the first time..."):
            run_ingestion()
    return True

if check_db():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about ILD..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                answer = ask_question(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

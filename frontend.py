import streamlit as st
import os

# --- IMPORT YOUR MODULES ---
from ingestion import main as run_ingestion 
from retrieval import retrieve_docs
from historybased import ask_question as get_answer

st.set_page_config(page_title="ILD Specialist Bot", page_icon="🫁")
st.title("🫁 ILD RAG Chatbot")

# --- STARTUP: ENSURE DB EXISTS ---
@st.cache_resource
def startup():
    if not os.path.exists("./chroma_db"):
        st.info("Building knowledge base...")
        run_ingestion()
    return True

if startup():
    # Chat History Setup
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Ask about ILD..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Use the new retrieve_docs function
            with st.expander("Show relevant document snippets"):
                context = retrieve_docs(prompt)
                st.write(context)
            
            # Get LLM Answer
            response = get_answer(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

import streamlit as st
import os
from ingestion import main as run_ingestion
from historybased import ask_question

st.set_page_config(page_title="ILD Specialist Bot", page_icon="🫁")
st.title("🫁 ILD RAG Expert")

# Initialize DB if it doesn't exist
if not os.path.exists("./chroma_db"):
    st.warning("Knowledge base not found. Initializing...")
    run_ingestion()
    st.rerun()

# Initialize History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about ILD..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # The historybased.py function now handles adding this to history internally
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_question(prompt)
            st.markdown(answer)
    
    # Save to history AFTER the UI renders to keep it clean
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})

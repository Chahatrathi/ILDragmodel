import streamlit as st
import os
# This matches the 'def main():' in your corrected ingestion.py
from ingestion import main as run_ingestion 
from historybased import ask_question

st.set_page_config(page_title="ILD Specialist Bot", page_icon="🫁", layout="wide")

# Custom CSS to make the chat look professional
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🫁 ILD RAG Expert")
st.caption("Specialist AI for Interstitial Lung Disease Research")

# 1. Startup Logic: Ensure Vector DB exists
# We don't cache the actual ingestion call itself to ensure UI messages show up
if not os.path.exists("./chroma_db"):
    st.info("No knowledge base found. Initializing documents...")
    with st.spinner("Analyzing PDFs... This may take a minute."):
        run_ingestion()
    # Force a rerun to clear the "No knowledge base" message and start the chat
    st.rerun()

# 2. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Chat Input & Response Logic
if prompt := st.chat_input("Ask a question about ILD diagnostics, treatments, or research..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting ILD database..."):
            try:
                # Assuming ask_question handles the retrieval and LLM logic
                answer = ask_question(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")

import streamlit as st
import os
from historybased import ask_question as get_response
# Ensure your ingestion.py has a 'main()' function that processes the data
from ingestion import main as run_ingestion 

# --- PAGE CONFIG ---
st.set_page_config(page_title="ILD Chatbot", page_icon="🫁")
st.title("🫁 ILD RAG Chatbot")
st.markdown("Ask questions about Interstitial Lung Disease based on the uploaded dataset.")

# --- DATA INITIALIZATION ---
# This function runs once and caches the result so the app stays fast.
@st.cache_resource
def initialize_app():
    # 1. Check if the vector database already exists
    # Replace 'chroma_db' with the actual folder name your ingestion script creates
    if not os.path.exists("./chroma_db"):
        st.info("First-time setup: Building the knowledge base from your documents...")
        try:
            # 2. Run the ingestion logic
            run_ingestion()
            st.success("Knowledge base built successfully!")
        except Exception as e:
            st.error(f"Error during ingestion: {e}")
            return False
    return True

# Trigger initialization
app_ready = initialize_app()

# --- CHAT INTERFACE ---
if app_ready:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is ILD?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    # Call the RAG function from historybased.py
                    response = get_response(prompt)
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
else:
    st.error("The app could not be initialized. Please check your ingestion.py and dataset.")

import streamlit as st
import os

# --- IMPORT YOUR PROJECT MODULES ---
# Ensure these function names match exactly what is in your files
try:
    from ingestion import main as run_ingestion 
    from retrieval import retrieve_docs  # Assuming this returns relevant chunks
    from historybased import ask_question as get_answer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ILD Medical Chatbot", page_icon="🫁", layout="centered")
st.title("🫁 ILD RAG Specialist")
st.markdown("---")

# --- STEP 1: INITIALIZE DATASET (Runs once) ---
@st.cache_resource
def startup_checks():
    # If the database folder doesn't exist, run ingestion
    if not os.path.exists("./chroma_db"):
        with st.status("Initializing Knowledge Base...", expanded=True) as status:
            st.write("Processing ILD documents...")
            run_ingestion()
            status.update(label="Knowledge Base Ready!", state="complete", expanded=False)
    return True

if startup_checks():
    # --- STEP 2: CHAT HISTORY SETUP ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- STEP 3: USER INTERACTION ---
    if prompt := st.chat_input("Ask about ILD symptoms, diagnosis, or treatments..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Optional: Show the retrieval process
            with st.status("Searching research papers...", expanded=False):
                context_docs = retrieve_docs(prompt)
                st.write(context_docs)
            
            # Final Answer generation
            with st.spinner("Formulating response..."):
                full_response = get_answer(prompt)
                st.markdown(full_response)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.error("System failed to initialize. Check your logs in the 'Manage App' section.")

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st

def run_ingestion():
    # 1. Check and create documents directory
    docs_path = "./documents"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        st.error(f"Directory '{docs_path}' was missing and has been created. Please upload PDFs to it.")
        return

    # 2. Load Documents
    # Note: glob="**/*.pdf" allows searching in subfolders as well
    loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    try:
        docs = loader.load()
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return

    # VALIDATION: Check if any documents were actually found
    if not docs:
        st.warning("No PDF documents found in the './documents' folder. Please add files and try again.")
        return

    # 3. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Second Validation: Ensure splitting resulted in chunks
    if not splits:
        st.error("Documents were loaded but no text chunks were created. Check if the PDFs are password protected or empty.")
        return

    # 4. Create Vector Store
    try:
        # Use st.secrets for the API key (ensure this is set in your Streamlit dashboard)
        api_key = st.secrets["OPENAI_API_KEY"]
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # This is where the crash was happening; now protected by logic above
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory="./chroma_db"
        )
        st.success(f"Successfully processed {len(docs)} documents into {len(splits)} chunks!")
        
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")

if __name__ == "__main__":
    run_ingestion()

import os
import time
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def main():
    docs_path = "./documents"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)

    loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    try:
        docs = loader.load()
    except Exception as e:
        st.error(f"Error reading PDF files: {e}")
        return

    if not docs:
        st.info("The documents folder is empty. Please upload PDFs via the sidebar.")
        return 

    # 1. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 2. Setup Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    # 3. Batch Processing to avoid Rate Limits (429)
    # We process 20 chunks at a time, then wait 5 seconds.
    batch_size = 20 
    vectorstore = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            status_text.text(f"Processing chunks {i} to {min(i + batch_size, len(splits))} of {len(splits)}...")
            
            if vectorstore is None:
                # First batch: Initialize the Chroma database
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
            else:
                # Subsequent batches: Add to existing database
                vectorstore.add_documents(batch)
            
            # Update progress
            progress_bar.progress(min((i + batch_size) / len(splits), 1.0))
            
            # 🛑 CRITICAL: The "Breath" - Wait to avoid hitting the 100 RPM limit
            time.sleep(5) 

        st.success(f"Success! Knowledge base built with {len(splits)} chunks.")
        status_text.empty()
        
    except Exception as e:
        st.error(f"Vector store error: {e}")

if __name__ == "__main__":
    main()

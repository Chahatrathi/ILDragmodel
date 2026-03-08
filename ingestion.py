import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def main():
    docs_path = "./documents"
    
    # 1. Ensure directory exists
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        st.info(f"Created '{docs_path}' folder. Please upload your PDFs there.")
        return

    # 2. Load Documents
    loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    try:
        docs = loader.load()
    except Exception as e:
        st.error(f"Error reading PDF files: {e}")
        return

    # Check if folder is empty to prevent ValueError
    if not docs:
        st.warning("No PDF documents found. Please add files to the /documents folder and refresh.")
        st.stop() 

    # 3. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 4. Create Vector Store
    try:
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory="./chroma_db"
        )
        st.success(f"Ingestion complete! Processed {len(docs)} files.")
    except Exception as e:
        st.error(f"Vector store error: {e}")

if __name__ == "__main__":
    main()

import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    try:
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        # Clear old DB if exists to prevent duplicate data
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory="./chroma_db"
        )
        st.success(f"Knowledge base updated! Processed {len(docs)} files.")
    except Exception as e:
        st.error(f"Vector store error: {e}")

if __name__ == "__main__":
    main()

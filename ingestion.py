import os
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
    docs = loader.load()

    if not docs:
        st.info("Please upload PDFs via the sidebar.")
        return 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    try:
        # Use Google Gemini Embeddings (Free Tier)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory="./chroma_db"
        )
        st.success(f"Knowledge base updated with Gemini!")
    except Exception as e:
        st.error(f"Vector store error: {e}")

if __name__ == "__main__":
    main()

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st

def main():
    # 1. Load Documents
    if not os.path.exists("./documents"):
        os.makedirs("./documents")
        print("Please put your ILD PDFs in the 'documents' folder.")
        return

    loader = DirectoryLoader("./documents", glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Create Vector Store
    # We use st.secrets to get the key from Streamlit Cloud
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("Successfully created chroma_db!")

if __name__ == "__main__":
    main()

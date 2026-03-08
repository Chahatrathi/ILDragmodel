import os
import time
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def main():
    persist_dir = "/tmp/chroma_db"
    if not os.path.exists("./documents"): os.makedirs("./documents")

    loader = DirectoryLoader("./documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    # Clean and Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = [s for s in text_splitter.split_documents(docs) if len(s.page_content.strip()) > 10]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    # 🛑 Clear old DB to prevent dimension mismatch
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)

    batch_size = 10
    vectorstore = None
    progress = st.progress(0)

    try:
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            if vectorstore is None:
                vectorstore = Chroma.from_documents(documents=batch, embedding=embeddings, persist_directory=persist_dir)
            else:
                vectorstore.add_documents(batch)
            progress.progress(min((i + batch_size) / len(splits), 1.0))
            time.sleep(12) # Safe throttle for Free Tier
        st.success("ILD Database Built!")
    except Exception as e:
        st.error(f"Ingestion Error: {e}")

if __name__ == "__main__":
    main()

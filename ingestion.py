import os
import time
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def main():
    docs_path = "./documents"
    if not os.path.exists(docs_path): os.makedirs(docs_path)

    loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        st.info("The documents folder is empty. Please upload PDFs via the sidebar.")
        return 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 🛑 DELETE old DB to avoid dimension errors
    if os.path.exists("./chroma_db"):
        import shutil
        shutil.rmtree("./chroma_db")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    # Throttling to stay under Free Tier limits
    batch_size = 15
    vectorstore = None
    progress = st.progress(0)

    try:
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            if vectorstore is None:
                vectorstore = Chroma.from_documents(documents=batch, embedding=embeddings, persist_directory="./chroma_db")
            else:
                vectorstore.add_documents(batch)
            progress.progress(min((i + batch_size) / len(splits), 1.0))
            time.sleep(10) # Safe for Free Tier
        st.success("Knowledge base ready!")
    except Exception as e:
        st.error(f"Vector store error: {e}")

if __name__ == "__main__":
    main()

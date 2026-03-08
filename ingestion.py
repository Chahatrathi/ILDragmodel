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
    
    try:
        docs = loader.load()
    except Exception as e:
        st.error(f"Error loading PDFs: {e}")
        return

    # 1. Clean and Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    raw_splits = text_splitter.split_documents(docs)
    
    # 2. Filter out empty or extremely short chunks that confuse the model
    splits = [s for s in raw_splits if len(s.page_content.strip()) > 10]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    batch_size = 10 # Smaller batches are safer for 500 errors
    vectorstore = None
    progress_bar = st.progress(0)

    try:
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch, 
                        embedding=embeddings, 
                        persist_directory=persist_dir
                    )
                else:
                    vectorstore.add_documents(batch)
            except Exception as batch_error:
                # If a specific batch fails, we log it and keep going
                st.warning(f"Skipping a small batch due to a server error. Continuing...")
                continue
            
            progress_bar.progress(min((i + batch_size) / len(splits), 1.0))
            time.sleep(8) 

        st.success("Knowledge base built! (Some problematic chunks may have been skipped).")
    except Exception as e:
        st.error(f"Critical Vector Store Error: {e}")

if __name__ == "__main__":
    main()

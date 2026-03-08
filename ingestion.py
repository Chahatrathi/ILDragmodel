import os
import time
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def main():
    persist_dir = "/tmp/chroma_db"
    
    # 1. Load Documents
    loader = DirectoryLoader("./documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    # 2. Optimized Splitting (Larger chunks = fewer API calls)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    # Only keep substantial content to save quota
    splits = [s for s in text_splitter.split_documents(docs) if len(s.page_content.strip()) > 50]

    st.write(f"Total chunks to process: {len(splits)}")
    if len(splits) > 1000:
        st.warning("⚠️ Warning: Your PDF is too large for the Free Tier daily limit (1000). Only the first 1000 chunks will be processed.")
        splits = splits[:950] # Safety margin

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    # 🛑 Force clean start for fresh indexing
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)

    batch_size = 5 # Very small batches for reliability
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
            # Wait 15 seconds between batches to stay under RPM limits
            time.sleep(15) 
            
        st.success("ILD Database Built Successfully!")
    except Exception as e:
        if "429" in str(e):
            st.error("🚨 Daily Quota Exceeded! Google's Free Tier only allows 1000 embeddings per day. You will need to wait 24 hours or use a different API Key.")
        else:
            st.error(f"Ingestion Error: {e}")

if __name__ == "__main__":
    main()

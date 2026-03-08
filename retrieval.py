import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st

def retrieve_docs(query):
    persist_directory = "./chroma_db"
    
    if not os.path.exists(persist_directory):
        return "Knowledge base not found."

    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # Get top 3 snippets
    results = vectorstore.similarity_search(query, k=3)
    
    formatted_context = ""
    for i, doc in enumerate(results):
        source = doc.metadata.get('source', 'Unknown')
        formatted_context += f"**Source {i+1} ({source}):**\n{doc.page_content}\n\n---\n"
    
    return formatted_context

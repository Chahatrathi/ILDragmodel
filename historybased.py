import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from retrieval import retrieve_docs

def ask_question(query):
    # 1. Get Context from Retrieval
    context = retrieve_docs(query)
    
    # 2. Initialize OpenAI
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    # 3. Build the Prompt
    messages = [
        SystemMessage(content="You are a medical assistant specializing in Interstitial Lung Disease (ILD). Use the provided context to answer questions accurately."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]

    # 4. Get Response
    response = llm.invoke(messages)
    return response.content

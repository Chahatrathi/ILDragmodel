import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def ask_question(user_query):
    # Using the Stable 2026 Preview Alias
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.3
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history, formulate a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an ILD specialist. Use the context to answer: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    chat_history = []
    for msg in st.session_state.messages:
        role_class = HumanMessage if msg["role"] == "user" else AIMessage
        chat_history.append(role_class(content=msg["content"]))

    result = rag_chain.invoke({"input": user_query, "chat_history": chat_history})
    return result["answer"]

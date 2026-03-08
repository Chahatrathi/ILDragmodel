import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def ask_question(user_query):
    # 1. Setup Models (Gemini 1.5 Flash is the 2026 workhorse)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    # 2. History-aware logic (Reformulates user query based on past chat)
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history, formulate a standalone question from the user's latest input."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # 3. Specialist Persona
    system_prompt = (
        "You are an expert Interstitial Lung Disease (ILD) specialist. "
        "Answer the question using ONLY the retrieved context below. "
        "If you don't know, state that clearly. Be professional and concise.\n\n"
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # 4. Map Streamlit state to LangChain Messages
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # 5. Get Answer
    result = rag_chain.invoke({"input": user_query, "chat_history": chat_history})
    return result["answer"]

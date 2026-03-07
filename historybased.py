import os
import warnings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama # Updated import
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Suppress technical noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

load_dotenv()

# 1. Connect to your local document database
persistent_directory = "./chroma_db"
# Using the same local embedding model used during ingestion
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# 2. Set up local AI model (Ollama)
model = ChatOllama(model="llama3", temperature=0)

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question standalone using history
    if chat_history:
        # Ask AI to rewrite the question to be standalone
        rephrase_prompt = [
            SystemMessage(content="You are a helpful assistant. Rewrite the following user question to be a standalone search query based on the chat history. Respond ONLY with the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(rephrase_prompt)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt with Context
    context_text = "\n".join([f"- {doc.page_content}" for doc in docs])
    combined_input = f"""Based on the following documents and our history, answer this: {user_question}

    Documents:
    {context_text}

    Rule: Answer ONLY using the provided documents. If not found, say you don't have enough information.
    """
    
    # Step 4: Get the answer from Local LLM
    messages = [
        SystemMessage(content="You are a medical research assistant. Answer based ONLY on provided context."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Update history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"\nFinal Answer: {answer}")
    return answer

def start_chat():
    print("Ollama RAG Chat Initialized! Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not question.strip():
            continue
        ask_question(question)

if __name__ == "__main__":
    start_chat()
import os
import warnings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Suppress technical noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

load_dotenv()

persistent_directory = "./chroma_db"

# 1. Load Local Embeddings
print("--- Loading Local Embedding Model ---")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load the vector store
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

# 3. Initialize Local LLM (Ollama)
# Make sure you have run 'ollama run llama3' in your terminal first!
model = ChatOllama(model="llama3", temperature=0)

print("\n--- RAG System Ready ---")
print("Type your medical query below (or type 'exit' to quit):")

# 4. Interactive Loop
while True:
    query = input("\nUser Query: ")
    
    if query.lower() in ['exit', 'quit', 'q']:
        print("Exiting...")
        break
        
    if not query.strip():
        continue

    print("Searching documents...")

    # 5. Retrieve relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)

    # 6. Prepare Context
    context_text = "\n".join([f"- {doc.page_content}" for doc in relevant_docs])
    
    combined_input = f"""Use the following pieces of context to answer the user's question. 
If you don't know the answer based on the context, just say that you don't know.

Context:
{context_text}

Question: {query}
"""

    # 7. Generate Response
    try:
        messages = [
            SystemMessage(content="You are a medical research assistant. Answer based ONLY on provided context."),
            HumanMessage(content=combined_input),
        ]

        print("--- Generating Answer ---")
        result = model.invoke(messages)
        print("\nFinal Response:")
        print(result.content)
        print("-" * 50)

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
import os
import warnings

# 1. Suppress the Protobuf/TensorFlow version mismatch warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def main():
    # 2. Initialize the local embedding engine
    # This must match the model used in ingestion.py
    print("--- Loading Local ILD Research Model ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Connect to your local vector database
    persist_directory = "./chroma_db"
    
    if not os.path.exists(persist_directory):
        print(f"\n[ERROR] Database not found at {persist_directory}")
        print("Please run 'python3 ingestion.py' first.")
        return

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # 4. Set up the search parameters (Retrieving top 3 relevant sections)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("\n" + "="*50)
    print("ILD KNOWLEDGE RETRIEVAL SYSTEM ACTIVE")
    print("="*50)
    print("Ask about Epidemiology, Etiology, or Pathophysiology.")
    print("(Type 'exit' to quit)")

    while True:
        query = input("\nResearch Query: ")
        if query.lower() in ['exit', 'quit', 'q']:
            print("Closing system...")
            break

        print(f"Searching medical records for: '{query}'...")
        
        # 5. Execute similarity search
        results = retriever.invoke(query)

        if not results:
            print("No matching clinical data found.")
            continue

        # 6. Display technical findings with source tracking
        for i, doc in enumerate(results):
            source = doc.metadata.get('source', 'Unknown')
            print(f"\n[Result {i+1}] Source: {source}")
            print("-" * 30)
            print(doc.page_content.strip())
            print("-" * 30)

if __name__ == "__main__":
    main()
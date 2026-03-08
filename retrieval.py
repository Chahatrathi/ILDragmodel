import os
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

def retrieve_docs(query):
    """
    This function is called by frontend.py to find relevant snippets.
    """
    # 1. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Connect to Database
    persist_directory = "./chroma_db"
    
    if not os.path.exists(persist_directory):
        return "Error: Database not found. Please run ingestion first."

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # 3. Search
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)

    # 4. Format results for the web UI
    formatted_results = ""
    for i, doc in enumerate(results):
        source = doc.metadata.get('source', 'Unknown')
        formatted_results += f"**Source {i+1}: {source}**\n\n{doc.page_content.strip()}\n\n---\n"
    
    return formatted_results if formatted_results else "No matching clinical data found."

# Keep the main() block so you can still run it in terminal if you want
def main():
    print("--- ILD Retrieval System (Terminal Mode) ---")
    while True:
        query = input("\nResearch Query: ")
        if query.lower() in ['exit', 'quit', 'q']: break
        print(retrieve_docs(query))

if __name__ == "__main__":
    main()

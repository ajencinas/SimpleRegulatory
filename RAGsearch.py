# rag_console.py

import os
import time
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_vector_store(vector_store_dir="vectorstores", vector_store_name="all_documents_vector_store"):
    vector_store_path = os.path.join(vector_store_dir, vector_store_name)

    if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
        raise FileNotFoundError("❌ No FAISS vector store found. Please run vectorization first.")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.load_local(
        vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

def build_qa_chain(vector_store, stream=True):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.2,
        model="gpt-4",
        streaming=stream,
        callbacks=[StreamingStdOutCallbackHandler()] if stream else None
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

def run_rag_interface():
    """
    Console-based RAG interface.
    """
    print("📦 Loading vector store...\n")
    try:
        start = time.time()
        vector_store = load_vector_store()
        print(f"⏱️ Time to load FAISS: {time.time() - start:.2f} seconds")
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        return

    qa_chain = build_qa_chain(vector_store)

    print("\n📚 You can now ask questions about the documents in the store.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("❓ Your question: ").strip()
        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        try:
            start = time.time()
            result = qa_chain.invoke({"query": query})
            print(f"\n\n ⏱️ Time to invoke search and LLM: {time.time() - start:.2f} seconds")
            print(f"\n💡 Answer:\n{result['result']}\n")

            sources = result.get("source_documents", [])
            if sources:
                print("📂 Sources:")
                for doc in sources:
                    meta = doc.metadata
                    file_name = meta.get("file_name", "Unknown file")
                    page_number = meta.get("page_number", "?")
                    print(f"  - {file_name} (Page {page_number})")
            print()

        except Exception as e:
            print(f"⚠️ Error while processing query: {e}\n")

def main():
    run_rag_interface()

if __name__ == "__main__":
    main()

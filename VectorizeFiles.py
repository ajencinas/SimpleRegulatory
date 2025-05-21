import os
import asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def chunkpdf(file_path):
    """Processes a PDF file and returns chunks and basic metadata."""
    try:
        pdf_reader = PdfReader(file_path)
        page_texts = [page.extract_text() or '' for page in pdf_reader.pages]
        full_text = "".join(page_texts)

        if not full_text.strip():
            raise ValueError("No text found in the PDF file.")

        print(f"🔍 Debug: Total extracted text length for {file_path}: {len(full_text)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, separators=[".", "\n"]
        )
        chunks = text_splitter.split_text(full_text)

        merged_pages = [
            page_texts[i] + page_texts[i + 1] if i + 1 < len(page_texts) else page_texts[i]
            for i in range(len(page_texts))
        ]

        chunks_with_meta = []
        for chunk in chunks:
            page_number = next(
                (i + 1 for i, page_text in enumerate(page_texts) if chunk in page_text),
                next(
                    (i + 1 for i, merged_text in enumerate(merged_pages) if chunk in merged_text),
                    None
                )
            )
            chunks_with_meta.append({
                "chunk": chunk,
                "file": file_path,
                "page_number": page_number
            })

        print(f"✅ Total chunks created for {file_path}: {len(chunks)}")
        return chunks_with_meta

    except Exception as e:
        print(f"❌ Error while chunking {file_path}: {e}")
        return None

async def create_or_update_faiss_vector_db(all_documents, vector_store_path):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        if os.path.exists(os.path.join(vector_store_path, "index.faiss")):
            print("🔄 Existing FAISS index found. Loading and updating...")
            vector_store = FAISS.load_local(
                vector_store_path, embeddings, allow_dangerous_deserialization=True
            )
            vector_store.add_documents(all_documents)
        else:
            print("🆕 Creating new FAISS index...")
            vector_store = FAISS.from_documents(all_documents, embeddings)

        vector_store.save_local(vector_store_path)
        print(f"✅ Vector store saved to: {vector_store_path}")

    except Exception as e:
        print(f"❌ Error saving FAISS vector store: {e}")

async def vectorize_all_pdfs(input_dir="static/files", vector_store_dir="vectorstores", vector_store_name="all_documents_vector_store"):
    os.makedirs(vector_store_dir, exist_ok=True)
    vector_store_path = os.path.join(vector_store_dir, vector_store_name)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("❌ No PDF files found in the 'static/files/' folder.")
        return

    all_documents = []

    for filename in pdf_files:
        file_path = os.path.join(input_dir, filename)
        print(f"\n📄 Processing: {file_path}")
        chunks_with_meta = chunkpdf(file_path)
        if not chunks_with_meta:
            continue

        documents = [
            Document(
                page_content=chunk["chunk"],
                metadata={
                    "file_name": filename,
                    "page_number": chunk["page_number"],
                    "blob_url": chunk["file"]
                }
            )
            for chunk in chunks_with_meta
        ]
        all_documents.extend(documents)

    if all_documents:
        await create_or_update_faiss_vector_db(all_documents, vector_store_path)
    else:
        print("⚠️ No documents were parsed for vectorization.")

if __name__ == "__main__":
    asyncio.run(vectorize_all_pdfs())

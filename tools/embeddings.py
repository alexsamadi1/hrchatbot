import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tools.loaders import enrich_pdf_chunks, chunk_docx_with_metadata

# --- Load API Key ---
def get_openai_api_key():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("❌ OPENAI_API_KEY is not set. Please check your .env file or Streamlit secrets.")
    return key

# --- Load Vectorstore ---
def load_faiss_vectorstore(index_name, openai_api_key, index_dir="faiss_index"):
    path = Path(index_dir)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# --- Build and Save Combined Vectorstore ---
def build_combined_vectorstore(pdf_path: str, docx_path: str, index_path: str, api_key: str):
    print("📥 Enriching PDF handbook...")
    pdf_chunks = enrich_pdf_chunks(pdf_path)

    print("📥 Chunking DOCX orientation guide...")
    docx_chunks = chunk_docx_with_metadata(docx_path)

    all_chunks = pdf_chunks + docx_chunks
    print(f"✅ Total chunks: {len(all_chunks)}")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Vectorstore saved to: {index_path}/")
    return vectorstore  # ← ✅ ADD THIS

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# --- Build and Save Vectorstore ---
def build_vectorstore(pdf_path, index_path, api_key):
    print("🔍 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"✅ Loaded {len(pages)} pages")

    print("🔧 Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = splitter.split_documents(pages)
    print(f"📄 Created {len(docs)} chunks")

    # Optional: log relevant chunks
    for i, doc in enumerate(docs):
        if "remote" in doc.page_content.lower() or "telecommute" in doc.page_content.lower():
            print(f"\n🧠 [Chunk {i+1}] MATCH FOUND:\n{doc.page_content[:500]}\n---")

    print("📦 Embedding and saving FAISS index...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Vectorstore saved to '{index_path}/'")

# --- Run when executed directly ---
if __name__ == "__main__":
    PDF_PATH = "InnovimEmployeeHandbook.pdf"
    INDEX_PATH = "faiss_index_hr"

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        raise ValueError("❌ OPENAI_API_KEY not found in secrets.toml")

    # Clean old index
    if os.path.exists(INDEX_PATH):
        import shutil
        print("♻️ Deleting old FAISS index...")
        shutil.rmtree(INDEX_PATH)

    build_vectorstore(PDF_PATH, INDEX_PATH, api_key)

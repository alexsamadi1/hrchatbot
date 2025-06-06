import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Load API Key ---
def get_openai_api_key():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("❌ OPENAI_API_KEY is not set. Please check your .env file or Streamlit secrets.")
    return key

# --- Build and Save Combined Vectorstore (Reusable) ---
def build_vectorstore(
    pdf_path="InnovimEmployeeHandbook.pdf",
    docx_path="innovimnew.docx",
    index_path="faiss_index",
    api_key=None
):
    print("🔍 Loading PDF and DOCX...")
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()
    for doc in pdf_docs:
        doc.metadata["source"] = "employee_handbook"

    docx_loader = UnstructuredWordDocumentLoader(docx_path)
    docx_docs = docx_loader.load()
    for doc in docx_docs:
        doc.metadata["source"] = "orientation_guide"

    all_docs = pdf_docs + docx_docs
    print(f"✅ Loaded {len(all_docs)} total pages")

    print("🔧 Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = splitter.split_documents(all_docs)
    print(f"📄 Created {len(docs)} chunks")

    print("📦 Embedding and saving FAISS index...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key or get_openai_api_key())
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Ensure save path exists
    path = Path(index_path)
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(path)
    print(f"✅ Vectorstore saved to '{index_path}/'")

    return vectorstore

# Optional CLI use
if __name__ == "__main__":
    build_vectorstore(index_path="faiss_index_hr_combined")

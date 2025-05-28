import os
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

# --- Build and Save Combined Vectorstore ---
def build_combined_vectorstore(pdf_path, docx_path, index_path, api_key):
    # Load documents
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

    # Split into chunks
    print("🔧 Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = splitter.split_documents(all_docs)
    print(f"📄 Created {len(docs)} chunks")

    # Embed and save
    print("📦 Embedding and saving FAISS index...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Vectorstore saved to '{index_path}/'")

if __name__ == "__main__":
    PDF_PATH = "InnovimEmployeeHandbook.pdf"
    DOCX_PATH = "innovimnew.docx"
    INDEX_PATH = "faiss_index_hr_combined"

    api_key = get_openai_api_key()
    build_combined_vectorstore(PDF_PATH, DOCX_PATH, INDEX_PATH, api_key)

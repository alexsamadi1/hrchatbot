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

# --- Build and Save Combined Vectorstore ---
def build_vectorstore(
    pdf_path="docs/InnovimEmployeeHandbook.pdf",
    docx_path="docs/innovim_onboarding.docx",
    index_path="faiss_index",
    api_key=None
):
    print("🔍 Checking for existing FAISS index...")
    index_file = Path(index_path) / "index.faiss"

    embeddings = OpenAIEmbeddings(openai_api_key=api_key or get_openai_api_key())

    if index_file.exists():
        print(f"✅ Existing vectorstore found at '{index_path}/'. Loading...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    print("🚧 No index found. Building new vectorstore...")

    # --- Load PDF ---
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()
    for doc in pdf_docs:
        doc.metadata["source"] = "employee_handbook"

    # --- Load DOCX ---
    docx_loader = UnstructuredWordDocumentLoader(docx_path)
    docx_docs = docx_loader.load()
    for doc in docx_docs:
        doc.metadata["source"] = "orientation_guide"

    # --- Combine and Split ---
    all_docs = pdf_docs + docx_docs
    print(f"📄 Loaded {len(all_docs)} total documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = splitter.split_documents(all_docs)
    print(f"✂️ Split into {len(docs)} chunks")

    # --- Embed and Save ---
    print("💾 Saving FAISS index...")
    Path(index_path).mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)

    print(f"✅ Vectorstore built and saved to '{index_path}/'")
    return vectorstore

# --- Optional CLI ---
if __name__ == "__main__":
    build_vectorstore(index_path="faiss_index_hr_combined")


def rebuild_vectorstore_from_docs(docs_path="docs", faiss_path="faiss_index"):
    docs_path = Path(docs_path)
    all_docs = []

    for doc_file in docs_path.glob("*"):
        if doc_file.suffix == ".pdf":
            loader = PyPDFLoader(str(doc_file))
        elif doc_file.suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(str(doc_file))
        else:
            continue
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_path)
    return len(all_docs), len(chunks)

def rebuild_vectorstore_from_s3():
    import toml
    from tools.embeddings import build_combined_vectorstore

    secrets = toml.load(".streamlit/secrets.toml")

    pdf_path = "docs/InnovimEmployeeHandbook.pdf"
    docx_path = "docs/innovim_onboarding.docx"
    index_path = "faiss_index"
    api_key = secrets["OPENAI_API_KEY"]

    vectorstore = build_combined_vectorstore(pdf_path, docx_path, index_path, api_key)
    return vectorstore, len(vectorstore.docstore._dict), len(vectorstore.index_to_docstore_id)
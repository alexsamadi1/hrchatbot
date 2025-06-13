import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import boto3
import json
import hashlib

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

    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())

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
    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_path)
    return len(all_docs), len(chunks)


def rebuild_vectorstore_from_s3():
    print("🔄 Starting vectorstore rebuild from S3...")

    s3 = boto3.client("s3")
    bucket = "innovim-hr-docs-1"
    faiss_path = "faiss_index/index"
    processed_manifest_path = Path("faiss_index/processed_hashes.json")

    # Load previously processed hashes
    if processed_manifest_path.exists():
        with open(processed_manifest_path, "r") as f:
            processed_hashes = set(json.load(f))
    else:
        processed_hashes = set()

    response = s3.list_objects_v2(Bucket=bucket)
    if "Contents" not in response:
        print("❌ No documents found in S3.")
        return 0, 0

    docs = []
    new_hashes = []

    for obj in response["Contents"]:
        key = obj["Key"]
        if not key.endswith((".pdf", ".docx")):
            continue

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            s3.download_file(bucket, key, tmp_file.name)
            print(f"⬇️ Downloaded: {key}")

            with open(tmp_file.name, "rb") as f:
                file_bytes = f.read()
                file_hash = hashlib.md5(file_bytes).hexdigest()

            if file_hash in processed_hashes:
                print(f"⏭ Skipping duplicate content for: {key}")
                continue

            if key.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file.name)
            else:
                loader = UnstructuredWordDocumentLoader(tmp_file.name)

            loaded_docs = loader.load()
            print(f"📄 Loaded {len(loaded_docs)} pages from {key}")
            docs.extend(loaded_docs)
            new_hashes.append(file_hash)

    if not docs:
        print("✅ No new files to process.")
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"🔬 Created {len(chunks)} chunks total.")

    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs("faiss_index", exist_ok=True)
    vectorstore.save_local(faiss_path)

    # Save updated manifest
    processed_hashes.update(new_hashes)
    with open(processed_manifest_path, "w") as f:
        json.dump(list(processed_hashes), f)

    print(f"✅ Vectorstore saved to {faiss_path}")
    return len(new_hashes), len(chunks)

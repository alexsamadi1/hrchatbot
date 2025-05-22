from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def load_faiss_vectorstore_from_pdf(pdf_path: str, api_key: str):
    """Rebuilds FAISS vectorstore from PDF using LangChain and OpenAI embeddings."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

def build_prompt(query: str, documents: list) -> str:
    """Builds a prompt for the assistant using the retrieved context."""
    context = "\n\n".join([doc.page_content for doc in documents])
    return f"""You are an HR assistant. Use the following context to answer the question.
If the answer is not in the context, say you don’t know.

Context:
{context}

Question:
{query}
"""

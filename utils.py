from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def load_faiss_vectorstore(index_path: str, api_key: str):
    """
    Loads a FAISS vectorstore using OpenAI embeddings.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def build_prompt(query: str, documents: list) -> str:
    """
    Builds a structured prompt using the query and retrieved documents.
    If no documents are found, provides a fallback context.
    """
    context = "\n\n".join(doc.page_content for doc in documents) if documents else "[No relevant documents found.]"
    
    return (
        "You are a helpful assistant. Answer the question below using only the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

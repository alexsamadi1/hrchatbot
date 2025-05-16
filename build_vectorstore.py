from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import streamlit as st
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Load your document
pdf_path = "InnovimEmployeeHandbook.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)
docs = splitter.split_documents(pages)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create the FAISS vectorstore
vectorstore = FAISS.from_documents(docs, embeddings)

# Save it locally
vectorstore.save_local("faiss_index_hr")
print("✅ Vectorstore saved to 'faiss_index_hr/'")

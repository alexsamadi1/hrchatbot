import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def enrich_pdf_chunks(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    raw_pages = loader.load()
    enriched_chunks = []

    section_pattern = re.compile(r"\n?(\d{3,4}\s+[A-Z][^\n]{3,}|[A-Z][A-Za-z\s]+\n)")

    for page_num, page in enumerate(raw_pages):
        text = page.page_content
        matches = list(section_pattern.finditer(text))
        positions = [m.start() for m in matches]

        if not positions:
            enriched_chunks.append(Document(
                page_content=text.strip(),
                metadata={"source": f"employee_handbook_page_{page_num + 1}"}
            ))
            continue

        positions.append(len(text))  # end of last section

        for i in range(len(positions) - 1):
            chunk_text = text[positions[i]:positions[i + 1]].strip()
            title_line = chunk_text.split("\n")[0].strip()
            title = re.sub(r"[^\w\s:]", "", title_line)

            enriched_text = (
                f"SECTION: {title}\n"
                f"Keywords: vacation, PTO, benefits, remote work, telecommute, timecard, leave, supervisor, holiday, HR, policy.\n\n"
                f"{chunk_text}"
            )

            enriched_chunks.append(Document(
                page_content=enriched_text,
                metadata={
                    "source": f"employee_handbook_page_{page_num + 1}",
                    "section_title": title
                }
            ))

    return enriched_chunks

def chunk_docx_with_metadata(docx_path: str) -> list:
    loader = UnstructuredWordDocumentLoader(docx_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["source"] = "orientation_guide"
    return chunks

def load_faiss_vectorstore(index_name, openai_api_key, index_dir="faiss_index"):
    path = Path(index_dir)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(path, embeddings, index_name=index_name, allow_dangerous_deserialization=True)

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

def build_prompt(query: str, documents: list) -> str:
    context_blocks = []
    for doc in documents:
        title = doc.metadata.get("section_title", "Unknown Section")
        source = doc.metadata.get("source", "")
        block = f"[{title} | {source}]\n{doc.page_content}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are an Innovim HR assistant. Use only the following context from the official Innovim Employee Handbook and Orientation Guide to answer.

If the answer is not clearly in the context, say: "I couldn’t find that in the handbook. Please check with HR."

Context:
{context}

Question:
{query}
"""

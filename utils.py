from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import re

def load_faiss_vectorstore_from_pdf(pdf_path: str, api_key: str):
    """
    Rebuild FAISS vectorstore with improved section detection:
    - Detects both numeric (e.g., 508 Paid Time Off) and labeled (e.g., Telecommuting)
    - Enriches chunks with section titles and common HR keywords
    """
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
                metadata={"source": f"Page {page_num + 1}"}
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
                    "source": f"Page {page_num + 1}",
                    "section_title": title
                }
            ))

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(enriched_chunks, embeddings)

    return vectorstore

def build_prompt(query: str, documents: list) -> str:
    context_blocks = []
    for doc in documents:
        title = doc.metadata.get("section_title", "Unknown Section")
        source = doc.metadata.get("source", "")
        block = f"[{title} | {source}]\n{doc.page_content}"
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    return f"""You are an Innovim HR assistant. Use only the following context from the official Innovim Employee Handbook to answer.

If the answer is not clearly in the context, say: "I couldn’t find that in the handbook. Please check with HR."

Context:
{context}

Question:
{query}
"""


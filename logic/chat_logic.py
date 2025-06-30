import time
from openai import OpenAI
from typing import List, Tuple, Optional
from langchain.schema.document import Document

# --- Rerank using GPT ---
def rerank_with_gpt(query, chunks, client: OpenAI) -> Optional[str]:
    if not chunks:
        return None

    context_snippets = "\n\n".join([f"Chunk {i+1}:\n{chunk.page_content[:500]}" for i, chunk in enumerate(chunks)])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Based on the user's question and the provided chunks of handbook and onboarding text, "
                "choose the single chunk that most directly and fully answers the question. Only select a chunk if it clearly answers the question. "
                "If none of the chunks are clearly relevant, say so."
            )
        },
        {
            "role": "user",
            "content": f"User question: {query}\n\nChunks:\n{context_snippets}"
        }
    ]

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        content = response.choices[0].message.content.strip()
        return None if "none are clearly relevant" in content.lower() else content
    except Exception:
        return None

# --- Fallback Summarization ---
def summarize_fallback(query, chunks: List[Document], client: OpenAI) -> str:
    fallback_context = "\n\n".join([chunk.page_content[:500] for chunk in chunks[:3]])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant trained on Innovim's employee handbook and onboarding documents. "
                "Summarize a cautious answer using the text provided. If unclear, advise contacting HR. "
                "Never fabricate Innovim-specific policies."
            )
        },
        {
            "role": "user",
            "content": f"User question: {query}\n\nPartial content:\n{fallback_context}"
        }
    ]

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return response.choices[0].message.content.strip()
    except Exception:
        return "I'm not confident I can answer that directly. Please check the handbook or contact HR for guidance."

# --- Answer Revision ---
def revise_answer_with_gpt(question, draft_answer, client: OpenAI) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful HR assistant reviewing an answer. "
                "Make it clear, friendly, and helpful to the employee, but don't make up Innovim policy details."
            )
        },
        {
            "role": "user",
            "content": (
                f"{draft_answer}\n\n"
                f"Please improve this answer so it's clearer and more helpful to someone asking: \"{question}\""
            )
        }
    ]

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return response.choices[0].message.content.strip()
    except Exception:
        return draft_answer


# --- Unified Response Generator ---
def generate_response(
    query: str,
    docs: List[Document],
    client: OpenAI,
    user_profile: dict
) -> Tuple[str, str]:
    """
    Returns: (final_answer, source_title)
    """
    chunks = docs[:3]
    reranked = rerank_with_gpt(query, chunks, client)

    if reranked:
        system_prompt = (
            f"You are Innovim’s professional HR assistant. The user is a {user_profile['role']} "
            f"with {user_profile['tenure']} at the company.\n\n"
            "Your job is to clearly answer the user's HR question using the excerpt provided. "
            "If you're unsure, advise the user to contact HR."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User question: {query}\n\nRelevant excerpt:\n{reranked}"}
        ]
    else:
        fallback_context = "\n\n".join([chunk.page_content[:500] for chunk in chunks])
        messages = [
            {"role": "system", "content": (
                "You are a helpful HR assistant trained on Innovim documents. The question wasn’t answered clearly by any one excerpt, "
                "but here are some partial chunks. Summarize a helpful answer based on what you can."
            )},
            {"role": "user", "content": f"User question: {query}\n\nContext snippets:\n{fallback_context}"}
        ]

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    draft_answer = response.choices[0].message.content.strip()
    final_answer = revise_answer_with_gpt(query, draft_answer, client)

    source_doc = docs[0].metadata.get("source", "Unknown") if docs else "None"
    return final_answer, source_doc

def generate_answer(messages, client):
    """Call OpenAI and return the assistant's draft answer."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Failed to generate answer: {e}"
    
def build_messages(user_input, context_chunk, profile, fallback=False):
    role = profile.get("role", "employee")
    tenure = profile.get("tenure", "unknown tenure")

    if fallback:
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful HR assistant trained on Innovim documents. "
                    "The question wasn’t answered clearly by any one excerpt, but here are some partial chunks. "
                    "Summarize a helpful answer based on what you can.\n"
                    "If unsure, advise the user to contact HR."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User question: {user_input}\n\n"
                    f"Context snippets:\n{context_chunk}"
                )
            }
        ]
    else:
        source = context_chunk.get("source", "Unknown Document")
        page = context_chunk.get("page")
        source_citation = f"{source}, page {page}" if page else source

        return [
            {
                "role": "system",
                "content": (
                    f"You are Innovim’s professional HR assistant. The user is a {role} "
                    f"with {tenure} at the company.\n\n"
                    "Your job is to clearly answer the user's HR question using the excerpt provided. "
                    "Be helpful and professional. If you're unsure, advise the user to contact HR."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User question: {user_input}\n\n"
                    f"Relevant excerpt (from {source_citation}):\n\n{context_chunk['text']}"
                )
            }
        ]
import streamlit as st
from openai import OpenAI
from utils import build_prompt, load_faiss_vectorstore_from_pdf
import time

# --- Page Setup ---
st.set_page_config(page_title="Innovim HR Chatbot", page_icon="📘", layout="wide")

# --- Load Vectorstore (from PDF, with caching) ---
@st.cache_resource(show_spinner="Indexing HR Handbook...")
def get_vectorstore():
    return load_faiss_vectorstore_from_pdf("InnovimEmployeeHandbook.pdf", st.secrets["OPENAI_API_KEY"])

vectorstore = get_vectorstore()

# --- Set up OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Rerank Logic ---
def rerank_with_gpt(query, chunks, client):
    if not chunks:
        return None

    context_snippets = "\n\n".join([f"Chunk {i+1}:\n{chunk.page_content[:500]}" for i, chunk in enumerate(chunks)])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Based on the user's question and the provided chunks of handbook text, "
                "choose the single chunk that most directly and completely answers the question. "
                "Only select a chunk if it clearly answers the question. If none of the chunks are clearly relevant, say so."
            )
        },
        {
            "role": "user",
            "content": f"User question: {query}\n\nChunks:\n{context_snippets}\n\nWhich chunk best answers the question? Reply with the full content of the best chunk, or say 'none are clearly relevant.'"
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        content = response.choices[0].message.content.strip()

        if "none are clearly relevant" in content.lower():
            return None
        return content

    except Exception as e:
        return None

# --- Answer Refinement with Completion Logic ---
def revise_answer_with_gpt(question, draft_answer, client):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to both company policy and general HR knowledge. "
                "You are reviewing a draft answer about an HR policy question. If the answer is vague, incomplete, or confusing, "
                "you may revise it using general human reasoning and best practices in HR. You may clarify, add logical context, or expand. "
                "However, you must NOT fabricate Innovim-specific policy details that were not part of the original handbook context."
            )
        },
        {
            "role": "user",
            "content": f"User question: {question}\n\nDraft answer: {draft_answer}\n\nPlease revise this response to make it clearer, more complete, and helpful, while avoiding made-up policy claims."
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return draft_answer

# --- Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Meta Query Detector ---
def detect_meta_query(query):
    q = query.lower()
    return any(phrase in q for phrase in [
        "what can you do", "how do you work", "can i ask you",
        "what is this", "how can you help", "help me", "who are you",
        "hi", "hello"
    ])

# --- Sidebar ---
with st.sidebar:
    st.image("innovimvector.png", use_container_width=True)
    st.markdown("### 🤖 Innovim HR Chatbot")
    st.markdown("_Your internal assistant for fast, reliable HR answers._")
    st.markdown("---")
    st.markdown("### 🧭 How to Use")
    st.markdown("""
    • Type a question related to HR or PM policies  
    • The bot searches the internal handbook  
    • You'll get a helpful, sourced response
    """)
    st.markdown("### 📬 Need More Help?")
    if st.button("Contact HR"):
        st.markdown("_Please email **hr@innovim.com**_")
    st.markdown("### 🔄 Start Over")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown("### 💬 Feedback")
    st.markdown("[Share your feedback here](https://docs.google.com/forms/d/e/1FAIpQLSc31lOd_KRn9mpffhQNwuthyzh1b3KTSeMGpb12hdJQ5IT_hQ/viewform?usp=dialog)")
    st.markdown("---")
    st.markdown("<div style='font-size: 0.8rem; color: gray;'>Version 1.0 • Last updated May 2025</div>", unsafe_allow_html=True)

# --- Main Header ---
st.markdown("<h1 style='text-align: center;'>Innovim HR Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Your go-to assistant for HR policies, benefits, and employee questions.</p>", unsafe_allow_html=True)

# --- Sample Questions ---
examples = [
    "How many vacation days do I get?",
    "What’s the policy on remote work?",
    "How do I update my benefits info?"
]

with st.expander("💡 Try a sample question", expanded=False):
    for q in examples:
        if st.button(q, key=f"sample_{q}"):
            st.session_state["example_question"] = q

# --- Show Chat History ---
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        bubble = "user-bubble" if entry["role"] == "user" else "bot-bubble"
        st.markdown(f"<div class='chat-bubble {bubble}'>{entry['content']}</div>", unsafe_allow_html=True)

# --- Handle User Input ---
user_input = st.chat_input("Ask a question about HR policies, benefits, or employee resources…")

if "example_question" in st.session_state and not user_input:
    user_input = st.session_state.pop("example_question")

if user_input:
    st.chat_message("user").markdown(f"<div class='chat-bubble user-bubble'>{user_input}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Early return for general questions
    if detect_meta_query(user_input):
        meta_response = (
            "Hi! 👋 I'm Innovim’s internal HR assistant. I can help answer questions about policies, benefits, timekeeping, telework, and more — "
            "all based on our official employee handbook.\n\n"
            "Try asking something like:\n"
            "• How many vacation days do I get?\n"
            "• What’s the policy on remote work?\n"
            "• What happens if I forget to log my time?"
        )
        st.chat_message("assistant").markdown(f"<div class='chat-bubble bot-bubble'>{meta_response}</div>", unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": meta_response})
        st.stop()

    with st.spinner("Searching policies..."):
        results = vectorstore.similarity_search_with_score(user_input, k=5)
        docs = [doc for doc, score in results if score >= 0.25]

        best_chunk = rerank_with_gpt(user_input, docs, client)

        if not best_chunk:
            answer = "I couldn’t find a strong match in the handbook. Please try rephrasing or contact HR."
        else:
            messages = [
                {"role": "system", "content": (
                    "You are Innovim’s professional HR assistant. "
                    "Only use the provided handbook content to answer. "
                    "If unclear, say: 'I couldn’t find a specific policy. Please check with HR.'"
                )},
                {"role": "user", "content": f"User question: {user_input}\n\nContext:\n{best_chunk}"}
            ]
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                draft_answer = response.choices[0].message.content
                revised = revise_answer_with_gpt(user_input, draft_answer, client)
                answer = revised.replace("Revised answer:", "").strip()
            except Exception as e:
                answer = f"❌ OpenAI error: {str(e)}"

    st.chat_message("assistant").markdown(f"<div class='chat-bubble bot-bubble'>{answer}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

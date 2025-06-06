import streamlit as st
from openai import OpenAI
from utils import load_faiss_vectorstore
import time
import re

DEBUG = False  # Set to True to show debug outputs

# --- Page Setup ---
st.set_page_config(page_title="Innovim HR Chatbot", page_icon="📘", layout="wide")

# --- Global CSS Styling ---
st.markdown("""
<style>
.chat-bubble {
  margin: 0.5rem 0;
  padding: 0.75rem 1rem;
  border-radius: 1rem;
  display: inline-block;
  max-width: 90%;
  line-height: 1.5;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.user-bubble {
  background-color: #1B4F72;  /* navy blue */
  color: #ffffff;
  align-self: flex-end;
}

.bot-bubble {
  background-color: #F0F4F8;  /* soft white */
  color: #0B1724;
  align-self: flex-start;
}
    
.typing-dots::after {
  content: '';
  display: inline-block;
  animation: dots 1.2s steps(3, end) infinite;
}

@keyframes dots {
  0% { content: ''; }
  33% { content: '.'; }
  66% { content: '..'; }
  100% { content: '...'; }
}
</style>
""", unsafe_allow_html=True)

# --- User Onboarding (Role & Tenure Selection) ---
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

profile = st.session_state.user_profile

# --- User Onboarding (Improved Flow) ---
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

if "role_selection" not in st.session_state:
    st.session_state.role_selection = None

if "tenure_selection" not in st.session_state:
    st.session_state.tenure_selection = None

profile = st.session_state.user_profile

if "role" not in profile or "tenure" not in profile:
    st.markdown("## 👋 Welcome! Let’s get to know you first")

    st.session_state.role_selection = st.radio(
        "What's your role at Innovim?",
        ["Project Manager", "General Staff", "Executive", "Contractor or Consultant"],
        key="role_radio_updated"
    )

    st.session_state.tenure_selection = st.radio(
        "How long have you been with Innovim?",
        ["New Hire (0–30 days)", "1–6 Months", "6+ Months", "2+ Years"],
        key="tenure_radio_updated"
    )

    if st.button("✅ Continue"):
        profile["role"] = st.session_state.role_selection
        profile["tenure"] = st.session_state.tenure_selection
        st.success("You're all set! You can now start asking questions below 👇")
        st.rerun()
    else:
        st.stop()

# --- Load Vectorstore ---
@st.cache_resource(show_spinner="🔍 Loading vectorstore...")
def get_vectorstore():
    try:
        return load_faiss_vectorstore("index", st.secrets["OPENAI_API_KEY"], index_dir="faiss_index")
    except Exception as e:
       #st.warning("Vectorstore not found. Rebuilding it now…")
        from tools.build_combined_vectorstore import build_vectorstore
        vectorstore = build_vectorstore(
            pdf_path="InnovimEmployeeHandbook.pdf",
            docx_path="innovimnew.docx",
            index_path="faiss_index",
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        return vectorstore  # ✅ make sure this is not returning a bool (e.g. `return vectorstore is`)


# ✅ Now safe to load vectorstore
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
                "You are a helpful assistant. Based on the user's question and the provided chunks of handbook and onboarding text, "
                "choose the single chunk that most directly and fully answers the question. Only select a chunk if it clearly answers the question. "
                "If none of the chunks are clearly relevant, say so."
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
            return summarize_fallback(query, chunks, client)
        return content

    except Exception:
        return None
    
# --- Summarize Fallback ---

def summarize_fallback(query, chunks, client):
    fallback_context = "\n\n".join([chunk.page_content[:500] for chunk in chunks[:3]])  # top 3 chunks

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant trained on Innovim's employee handbook and onboarding documents. "
                "The user asked a question that wasn't answered clearly by a single chunk, but we’ve gathered related information. "
                "Using these, summarize a helpful, cautious response — and if the answer is uncertain, recommend the user contact HR. "
                "Never fabricate Innovim policy details."
            )
        },
        {
            "role": "user",
            "content": f"User question: {query}\n\nPartial content:\n{fallback_context}\n\nPlease provide the most helpful answer you can from this content."
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return "I'm not confident I can answer that directly. Please check the handbook or contact HR for guidance."
    
# --- Answer Refinement ---
def revise_answer_with_gpt(question, draft_answer, client):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to both Innovim's handbook and onboarding documents. "
                "You are reviewing a draft answer about an HR policy or employee process question. If the answer is vague, incomplete, or confusing, "
                "you may revise it using general human reasoning and best practices in HR. You may clarify, add logical context, or expand. "
                "However, you must NOT fabricate Innovim-specific policy details that were not part of the original documents."
            )
        },
        {
            "role": "user",
            "content": f"User question: {question}\n\nDraft answer: {draft_answer}\n\nPlease revise this response to make it clearer, more complete, and helpful, while avoiding made-up policy claims."
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content.strip().replace("Revised answer:", "").strip()
    except Exception:
        return draft_answer

# --- Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Meta Query Detector ---
def detect_meta_query(query):
    q = query.lower().strip()

    # Match only simple greeting or assistant-checking phrases
    meta_patterns = [
        r"^\s*hi\s*$",
        r"^\s*hello\s*$",
        r"^\s*who are you\??\s*$",
        r"^\s*what can you do\??\s*$",
        r"^\s*how do you work\??\s*$",
        r"^\s*can i ask you\??\s*$",
        r"^\s*help me\??\s*$",
        r"^\s*how can you help\??\s*$",
        r"^\s*what is this\??\s*$"
    ]

    return any(re.match(pattern, q) for pattern in meta_patterns)

# def detect_meta_query(query):
#     q = query.lower()
#     return any(phrase in q for phrase in [
#         "what can you do", "how do you work", "can i ask you",
#         "what is this", "how can you help", "help me", "who are you",
#         "hi", "hello"
#     ])


# --- Sidebar ---

with st.sidebar:
    st.image("innovimvector.png", use_container_width=True)

    st.markdown("## 🤖 Innovim HR Assistant")
    st.caption("_Your personal guide for Innovim HR policies & info._")

    st.markdown("### 🧭 Quick Start")
    st.markdown("Ask about:\n- PTO / Vacation\n- Remote work\n- Benefits updates\n- Time tracking")

    st.markdown("### 💬 Sample Questions")
    sample_questions = [
        "How many vacation days do I get?",
        "What’s the policy on remote work?",
        "How do I update my benefits info?"
    ]
    for i, q in enumerate(sample_questions):
        if st.button(f"💡 {q}", key=f"sample_q_{i}"):
            st.session_state["example_question"] = q

    st.markdown("---")

    st.markdown("### 📬 Need More Help?")
    st.markdown("Email [hr@innovim.com](mailto:hr@innovim.com)")

    if st.button("🔄 Start Over"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")

    st.markdown("### 💡 Feedback & Version")
    st.markdown("[📣 Submit Feedback](https://docs.google.com/forms/d/e/1FAIpQLSc31lOd_KRn9mpffhQNwuthyzh1b3KTSeMGpb12hdJQ5IT_hQ/viewform?usp=dialog)")
    st.markdown("<div style='font-size: 0.8rem; color: gray;'>🔒 Internal Use Only • Version 1.0 • Updated May 2025</div>", unsafe_allow_html=True)

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

# --- Empty State UX ---
if not st.session_state.chat_history and "example_question" not in st.session_state:
    with st.chat_message("assistant"):
        st.markdown("""
        <div class='chat-bubble bot-bubble'>
             Hi there! I’m your Innovim HR Assistant. You can ask me anything about:
            <ul>
                <li> Time tracking</li>
                <li> Vacation / PTO</li>
                <li> Remote work</li>
                <li> Benefits & forms</li>
            </ul>
            Just type your question below or click one of the samples to get started.
        </div>
        """, unsafe_allow_html=True)

# --- Show Chat History ---
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        bubble = "user-bubble" if entry["role"] == "user" else "bot-bubble"
        st.markdown(f"<div class='chat-bubble {bubble}'>{entry['content']}</div>", unsafe_allow_html=True)

# --- Handle User Input ---
user_input = st.chat_input("Ask a question about HR policies, benefits, or employee resources…")

if "example_question" in st.session_state and not user_input:
    user_input = st.session_state.pop("example_question")

# ✅ EXIT EARLY IF INPUT IS BLANK OR INVALID
if not user_input or not isinstance(user_input, str) or not user_input.strip():
    st.stop()

# from here on: guaranteed valid input
# Prevent blank or non-string inputs from continuing
if user_input:
    st.chat_message("user").markdown(f"<div class='chat-bubble user-bubble'>{user_input}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # # Meta query response
    # if detect_meta_query(user_input):
    #     with st.chat_message("assistant"):
    #         st.markdown(
    #             f"<div class='chat-bubble bot-bubble'>Hi! 👋 I'm Innovim’s internal HR assistant. I can help answer questions about policies, benefits, timekeeping, telework, and more — all based on our official employee handbook.<br><br>Try asking something like:<br>• How many vacation days do I get?<br>• What’s the policy on remote work?<br>• What happens if I forget to log my time?</div>",
    #             unsafe_allow_html=True
    #         )
    #     st.session_state.chat_history.append({"role": "assistant", "content": "Hi! 👋 I'm Innovim’s internal HR assistant..."})
    #     st.stop()

with st.spinner("Searching policies..."):
    with st.chat_message("assistant"):
        # Step 1: Show animated typing message first
        placeholder = st.empty()
        placeholder.markdown(
            "<div class='chat-bubble bot-bubble'>🤖 <span class='typing-dots'>Typing</span></div>",
            unsafe_allow_html=True
        )

        # Step 2: Run similarity search and reranking
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            st.stop()
        results = vectorstore.similarity_search_with_score(user_input, k=5)
        docs = [doc for doc, score in results if score >= 0.25]
        best_chunk = rerank_with_gpt(user_input, docs, client)

        # Step 3: If no match found
        if not best_chunk:
            answer = "I couldn’t find a strong match in the handbook. Please try rephrasing or contact HR."
        else:
            try:
                messages = [
                    {"role": "system", "content": (
                        f"You are Innovim’s professional HR assistant. The user is a {profile['role']} who has been with the company for {profile['tenure']}.\n"
                        "Use this context to tailor your answer whenever possible. "
                        "Only use the provided handbook content to answer. If unclear, say: 'I couldn’t find a specific policy. Please check with HR.'"
                    )},
                    {"role": "user", "content": f"User question: {user_input}\n\nContext:\n{best_chunk}"}
                ]

                # Step 4: Generate response
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                draft_answer = response.choices[0].message.content
                answer = revise_answer_with_gpt(user_input, draft_answer, client)

            except Exception as e:
                answer = f"❌ OpenAI error: {str(e)}"

        # Step 5: Replace placeholder with final answer
        placeholder.markdown(
            f"<div class='chat-bubble bot-bubble'>{answer}</div>",
            unsafe_allow_html=True
        )
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Debug tools
    if DEBUG:
        with st.expander("🛠 Debug Info"):
            st.write("Raw Retrieved Chunks", docs)
            st.write("Selected Chunk", best_chunk)
            st.write("Answer", answer)


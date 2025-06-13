import streamlit as st
from openai import OpenAI
from tools.embeddings import load_faiss_vectorstore
from tools.s3_utils import upload_file_to_s3
from tools.vectorstore_builder import rebuild_vectorstore_from_s3
from tools.log_utils import ensure_log_file_exists, log_query_to_csv
from tools.analytics_dashboard import show_analytics_dashboard
from pathlib import Path
import nltk
import uuid
import time
import re
import os

# --- Page Setup ---
st.set_page_config(page_title="Innovim HR Chatbot", page_icon="📘", layout="wide")
ensure_log_file_exists()

# Ensure all necessary NLTK resources are available
nltk_dependencies = [
    'punkt',
    'averaged_perceptron_tagger'
]

def ensure_nltk_resources(resources):
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'taggers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

ensure_nltk_resources(['punkt', 'averaged_perceptron_tagger'])


if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

DEBUG = False  # Set to True to show debug outputs


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

if "role_selection" not in st.session_state:
    st.session_state.role_selection = None

if "tenure_selection" not in st.session_state:
    st.session_state.tenure_selection = None

profile = st.session_state.user_profile

# --- If user not onboarded, show onboarding screen (no sidebar) ---
if "role" not in profile or "tenure" not in profile:
    st.markdown("## 👋 Welcome! Let’s get to know you first")

    st.session_state.role_selection = st.radio(
        "What's your role at Innovim?",
        ["Program Manager", "General Staff"],
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

if st.session_state.get("show_analytics", False):
    from tools.analytics_dashboard import show_analytics_dashboard
    show_analytics_dashboard()
    st.stop()


# --- Load Vectorstore ---
@st.cache_resource(show_spinner="🔍 Loading vectorstore...")
def get_vectorstore():
    try:
        vectorstore = load_faiss_vectorstore("index", st.secrets["OPENAI_API_KEY"])
        return vectorstore
    except Exception as e:
        st.warning(f"⚠️ Couldn’t load vectorstore from S3. Rebuilding... ({e})")
        vectorstore = rebuild_vectorstore_from_s3()
        return vectorstore
    
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
            model="gpt-3.5-turbo",
            messages=messages, 
            stream=True 
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

if "role" in profile and "tenure" in profile:
    with st.sidebar:
        # --- Logo ---
        st.image("assets/innovimvector.png", use_container_width=True)

        st.markdown("## 🤖 Innovim HR Assistant")
        st.caption("_Your personal guide for Innovim HR policies & info._")

        st.markdown("### 🧭 Quick Start")
        st.markdown("""
        Ask about:
        - PTO / Vacation  
        - Remote work  
        - Benefits updates  
        - Time tracking
        """)

        # --- Sample Questions ---
        with st.expander("💬 Sample Questions", expanded=False):
            sample_questions = [
                "How many vacation days do I get?",
                "What’s the policy on remote work?",
                "How do I update my benefits info?"
            ]
            for i, q in enumerate(sample_questions):
                if st.button(f"💡 {q}", key=f"sample_q_{i}"):
                    st.session_state["example_question"] = q

        # --- Admin Upload Tools ---
        with st.expander("🔒 Admin Upload Tools", expanded=False):
            admin_code = st.text_input("Enter admin code", type="password")

            # --- Grant access if correct
            if admin_code == st.secrets["ADMIN_CODE"]:
                if not st.session_state.is_admin:
                    st.success("✅ Admin access granted")
                st.session_state.is_admin = True

            # --- Show upload tools only if admin verified
            if st.session_state.get("is_admin", False):
                uploaded_file = st.file_uploader("📤 Upload HR document", type=["pdf", "docx"])
                if uploaded_file:
                    if "last_uploaded_file" not in st.session_state:
                        st.session_state.last_uploaded_file = None

                    if uploaded_file.name != st.session_state.last_uploaded_file:
                        unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
                        try:
                            upload_file_to_s3(uploaded_file, unique_filename, st.secrets["S3_DOCS_BUCKET"])
                            st.success(f"✅ Uploaded: {uploaded_file.name}")

                            with st.spinner("🔄 Rebuilding knowledge base..."):
                                doc_count, chunk_count = rebuild_vectorstore_from_s3()
                                st.success(f"📚 Vectorstore rebuilt from {doc_count} docs ({chunk_count} chunks)")

                            st.session_state.last_uploaded_file = uploaded_file.name
                            st.cache_resource.clear()
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Upload failed: {e}")
                    else:
                        st.info("ℹ️ This file was already uploaded in this session.")
        # ✅ Admin-only button to open Analytics Dashboard
        if st.session_state.get("is_admin", False):
            st.markdown("---")
            st.markdown("### 📊 Admin Tools")
            if st.sidebar.button("📊 Open Analytics Dashboard"):
                st.session_state.show_analytics = True

        st.markdown("---")

        # --- Help & Reset ---
        st.markdown("### 📬 Need Help?")
        st.markdown("[Email HR](mailto:hr@innovim.com)")

        if st.button("🔄 Start Over"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")

        # --- Footer ---
        st.markdown("### 💡 Feedback")
        st.markdown("[📣 Submit Feedback](https://docs.google.com/forms/d/e/1FAIpQLSc31lOd_KRn9mpffhQNwuthyzh1b3KTSeMGpb12hdJQ5IT_hQ/viewform?usp=dialog)")
        st.markdown("<div style='font-size: 0.8rem; color: gray;'>🔒 Internal • v1.0 • Updated May 2025</div>", unsafe_allow_html=True)

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
        # Step 1: Typing placeholder
        placeholder = st.empty()
        placeholder.markdown(
            "<div class='chat-bubble bot-bubble'>🤖 <span class='typing-dots'>Typing</span></div>",
            unsafe_allow_html=True
        )

        # Step 2: Search & rerank
        results = vectorstore.similarity_search_with_score(user_input, k=3)
        docs = [doc for doc, score in results if score >= 0.25]
        best_chunk = rerank_with_gpt(user_input, docs, client)

        # Step 3: Handle weak matches
        if not best_chunk:
            answer = "I couldn’t find a strong match in the handbook. Please try rephrasing or contact HR."
            placeholder.markdown(
                f"<div class='chat-bubble bot-bubble'>{answer}</div>",
                unsafe_allow_html=True
            )
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            log_query_to_csv(user_input, answer)
            st.stop()

        # Step 4: Generate GPT answer (no streaming)
        messages = [
            {"role": "system", "content": (
                f"You are Innovim’s professional HR assistant. The user is a {profile['role']} who has been with the company for {profile['tenure']}.\n"
                "Use this context to tailor your answer whenever possible. "
                "Only use the provided handbook content to answer. If unclear, say: 'I couldn’t find a specific policy. Please check with HR.'"
            )},
            {"role": "user", "content": f"User question: {user_input}\n\nContext:\n{best_chunk}"}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        draft_answer = response.choices[0].message.content.strip()

        # Refine and animate
        answer = revise_answer_with_gpt(user_input, draft_answer, client)
        lines = re.split(r'(?<=[.!?])\s+', answer)
        displayed = ""
        for line in lines:
            displayed += line + " "
            placeholder.markdown(
                f"<div class='chat-bubble bot-bubble'>{displayed.strip()}▌</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.8)

        placeholder.markdown(
            f"<div class='chat-bubble bot-bubble'>{displayed.strip()}</div>",
            unsafe_allow_html=True
        )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})




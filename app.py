import streamlit as st
from openai import OpenAI
from utils import build_prompt, load_faiss_vectorstore_from_pdf

# --- Page Setup ---
st.set_page_config(page_title="Innovim HR Chatbot", page_icon="📘", layout="wide")

# --- Load Vectorstore (from PDF, with caching) ---
@st.cache_resource(show_spinner="Indexing HR Handbook...")
def get_vectorstore():
    return load_faiss_vectorstore_from_pdf("InnovimEmployeeHandbook.pdf", st.secrets["OPENAI_API_KEY"])

vectorstore = get_vectorstore()

# --- Set up OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Custom Styles ---
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .chat-bubble {
            padding: 1rem;
            border-radius: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            max-width: 90%;
        }
        .user-bubble {
            background-color: #2B2B2B;
            color: white;
        }
        .bot-bubble {
            background-color: #1E1E1E;
            border: 1px solid #444;
            color: #eee;
        }
    </style>
""", unsafe_allow_html=True)

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

    # Build FAISS search query
    if len(st.session_state.chat_history) >= 2:
        last_bot = st.session_state.chat_history[-2]["content"]
        faiss_query = f"{last_bot}\n\n{user_input}"
    else:
        faiss_query = user_input

    with st.spinner("Thinking..."):
        docs = vectorstore.similarity_search(faiss_query, k=3)
        prompt = build_prompt(user_input, docs)

        if not docs:
            answer = "I couldn't find anything in the handbook for that. Please reach out to HR for clarification."
        else:
            messages = [
                {"role": "system", "content": "You are Innovim’s professional assistant, trained on the internal HR handbook. Use only the provided context. If unsure, recommend contacting HR."},
                {"role": "user", "content": prompt}
            ]
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"❌ OpenAI error: {str(e)}"

    st.chat_message("assistant").markdown(f"<div class='chat-bubble bot-bubble'>{answer}</div>", unsafe_allow_html=True)
    with st.expander("📄 View Sources"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Source {i+1}:** {doc.page_content[:500]}...")
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

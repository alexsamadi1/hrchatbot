import streamlit as st
from openai import OpenAI
from utils import load_faiss_vectorstore
from config import INDEX_PATH
from utils import build_prompt

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Page Setup ---
st.set_page_config(page_title="Innovim PM Chatbot", page_icon="📘", layout="wide")

# --- OpenAI Client ---
vectorstore = load_faiss_vectorstore(INDEX_PATH, st.secrets["OPENAI_API_KEY"])

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

#--- Sidebar ---
# --- Sidebar ---
with st.sidebar:
    # --- Scoped Sidebar Styles ---
    st.markdown("""
        <style>
        .sidebar-center {
            text-align: center;
        }
        .sidebar-footer {
            margin-top: 2rem;
            font-size: 0.8rem;
            color: gray;
            text-align: center;
        }
        .sidebar-button {
            display: flex;
            justify-content: center;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-center'>", unsafe_allow_html=True)

    # --- Logo ---
    st.image("innovimvector.png", use_container_width=True)


    # --- Branding ---
    st.markdown("### 🤖 Innovim HR Chatbot")
    st.markdown("_Your internal assistant for fast, reliable HR answers._", unsafe_allow_html=True)

    st.markdown("---")

    # --- How to Use ---
    st.markdown("### 🧭 How to Use")
    st.markdown("""
    • Type a question related to HR or PM policies  
    • The bot searches the internal handbook  
    • You'll get a helpful, sourced response
    """)

    # --- Disclaimer ---
    st.markdown("<small>⚠️ This chatbot is for informational purposes only.<br>For official HR decisions, please consult HR directly.</small>", unsafe_allow_html=True)

    st.markdown("---")

    # --- Contact HR ---
    st.markdown("### 📬 Need More Help?")
    st.markdown("<div class='sidebar-button'>", unsafe_allow_html=True)
    if st.button("Contact HR"):
        st.markdown("_Please email **hr@innovim.com**_", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Feedback ---
    st.markdown("### 💬 Feedback")
    st.markdown("Have suggestions or issues?")
    st.markdown("[Click here to share feedback](https://docs.google.com/forms/d/e/1FAIpQLSc31lOd_KRn9mpffhQNwuthyzh1b3KTSeMGpb12hdJQ5IT_hQ/viewform?usp=dialog)")

    st.markdown("---")

    # --- Clear Chat ---
    st.markdown("### 🔄 Start Over")
    st.markdown("Reset the conversation and start fresh.")
    st.markdown("<div class='sidebar-button'>", unsafe_allow_html=True)
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("<div class='sidebar-footer'>Version 1.0 • Last updated May 2025</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center;'>Innovim HR Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Your go-to assistant for HR policies, benefits, and employee questions.</p>", unsafe_allow_html=True)

# --- Sample Questions ---
examples = [
    "How many vacation days do I get?",
    "What’s the policy on remote work?",
    "How do I update my benefits info?"
]

with st.expander("💡 Need help? Try one of these sample questions:", expanded=False):
    for q in examples:
        if st.button(q, key=f"sample_{q}"):
            st.session_state["example_question"] = q

# --- Show Chat History ---
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        bubble = "user-bubble" if entry["role"] == "user" else "bot-bubble"
        st.markdown(f"<div class='chat-bubble {bubble}'>{entry['content']}</div>", unsafe_allow_html=True)

# --- User Input ---
user_input = st.chat_input("Ask about PM responsibilities, workflows, or policies...")

if "example_question" in st.session_state and not user_input:
    user_input = st.session_state.pop("example_question")

# --- Handle New Message ---
if user_input:
    st.chat_message("user").markdown(f"<div class='chat-bubble user-bubble'>{user_input}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Build smart FAISS query: combine last assistant message if available
    if len(st.session_state.chat_history) >= 2:
        last_bot = st.session_state.chat_history[-2]["content"]
        faiss_query = f"{last_bot}\n\n{user_input}"
    else:
        faiss_query = user_input

    with st.spinner("Thinking..."):
        # Get handbook chunks
        docs = vectorstore.similarity_search(faiss_query, k=3)
        prompt = build_prompt(user_input, docs)
        if not docs:
            answer = "I'm sorry, I couldn't find anything in the handbook that addresses that. You may want to reach out to HR for clarification."
        else:
            # Prepare conversation context
            messages = [
            {"role": "system", "content": "You are Innovim’s professional assistant, trained on the internal project management handbook. Use only the provided context to answer questions clearly, helpfully, and respectfully. If the answer isn’t available in the context, respond politely and suggest reaching out to HR."},
            {"role": "user", "content": prompt}
                    ]
            # Generate answer
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"❌ Error from OpenAI: {str(e)}"

    # Show and store assistant response
    st.chat_message("assistant").markdown(f"<div class='chat-bubble bot-bubble'>{answer}</div>", unsafe_allow_html=True)
    with st.expander("📄 View Sources"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}:** {doc.page_content[:500]}...")
    st.session_state.chat_history.append({"role": "assistant", "content": answer})



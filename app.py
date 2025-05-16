import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# --- Page Setup ---
st.set_page_config(page_title="Innovim PM Chatbot", page_icon="📘", layout="wide")

# --- Load Vectorstore ---
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
vectorstore = FAISS.load_local("faiss_index_hr", embeddings, allow_dangerous_deserialization=True)

# --- OpenAI Client ---
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
    st.markdown("""
        <h4 style="margin-top: 1rem;">How to Use</h4>
        <p style="font-size: 0.92rem; line-height: 1.6;">
            Ask questions about Innovim’s PM policies.<br>
            This chatbot is trained on the internal handbook and will respond using relevant sections.
        </p>
        <hr style="border: 0.5px solid #444;">
        <p style="font-size: 0.85rem; font-style: italic; color: #CCCCCC;">
            For official HR inquiries, please contact HR directly.
        </p>
        <div style='text-align: center; font-size: 0.8rem; color: #888; margin-top: 3rem;'>
            Built by Innovim · Version 1.0
        </div>
    """, unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center;'>Innovim PM Chatbot (MVP)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask any question related to Innovim’s project management policies and guidelines.</p>", unsafe_allow_html=True)

# --- Sample Questions ---
examples = [
    "How do I request PTO?",
    "Who approves scope changes?",
    "How are milestones tracked?"
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
        context = "\n\n".join([doc.page_content for doc in docs])

        if not docs:
            answer = "I'm sorry, I couldn't find anything in the handbook that addresses that. You may want to reach out to HR for clarification."
        else:
            # Prepare conversation context
            messages = [
                {"role": "system", "content": "You are Innovim’s professional assistant, trained on the internal project management handbook. Use only the provided context to answer questions clearly, helpfully, and respectfully. If the answer isn’t available in the context, respond politely and suggest reaching out to HR."},
                {"role": "system", "content": f"Handbook context:\n\n{context}"}
            ]

            for m in st.session_state.chat_history[-6:]:
                messages.append({"role": m["role"], "content": m["content"]})

            messages.append({"role": "user", "content": user_input})

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
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

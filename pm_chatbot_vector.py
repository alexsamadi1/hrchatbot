import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import pickle

# Load the vector store
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
vectorstore = FAISS.load_local("faiss_index_hr", embeddings, allow_dangerous_deserialization=True)


# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Define Streamlit app
st.set_page_config(page_title="Innovim PM Handbook Chatbot")
st.title("📘 PM Handbook Chatbot")
st.write("Ask anything from the Innovim PM Handbook:")

# Input
query = st.chat_input("Ask your HR/PM question here...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    # Run similarity search and GPT response
    docs = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Innovim’s helpful assistant trained on the PM handbook. Keep responses concise and professional."},
                {"role": "user", "content": f"Answer the question based on the following handbook context:\n\n{context}\n\nQuestion: {query}"}
            ]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"❌ Error from OpenAI:\n\n{str(e)}"

    # Display answer
    with st.chat_message("assistant"):
        st.markdown(answer)

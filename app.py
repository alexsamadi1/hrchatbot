import streamlit as st
from openai import OpenAI
from datetime import datetime
import pandas as pd

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Define system prompt to guide assistant behavior
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Innovim's analytics assistant. "
        "You help users interpret monitoring data, anomaly detection results, and report summaries. "
        "Always respond clearly and concisely using plain language."
    )
}

# Initialize Streamlit app UI
st.set_page_config(page_title="Innovim Chatbot (MVP Prototype)")
st.title("Innovim Chatbot (MVP Prototype)")
st.write("Ask a question and receive analytics-driven assistance.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_PROMPT]

# Display chat history
for message in st.session_state.messages[1:]:  # skip system prompt for UI
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
with st.expander("💡 Example Questions"):
    if st.button("📊 Show me a usage summary"):
        prompt = "Show me a usage summary"
    elif st.button("🔍 Were there any anomalies in the last 7 days?"):
        prompt = "Were there any anomalies in the last 7 days?"
    elif st.button("📈 How is the system performing this week?"):
        prompt = "How is the system performing this week?"
    else:
        prompt = None
user_input = st.chat_input("Ask something about the data...")
if user_input:
    prompt = user_input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    st.markdown(prompt)
    try:
        # Simulate custom logic for common analytics prompts
        lowered = prompt.lower()
        if "anomalies" in lowered or "last 7 days" in lowered:
            answer = (
                "📈 From May 9 to May 15, no critical anomalies were detected. "
                "There was a minor spike in API errors on May 12 due to a system update.\n\n"
                "_(This response simulates what would be returned from an Elasticsearch query.)_"
            )
        elif "summary" in lowered or "usage" in lowered:
            answer = (
                "🗂️ Weekly usage summary:\n- Total API calls: 12,457\n"
                "- Error rate: 0.3%\n- Uptime: 99.98%\n\n"
                "_(Simulated response — once integrated, this will be powered by real data.)_"
            )
        else:
            # Default GPT-4 response
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.messages
            )
            answer = response.choices[0].message.content

    except Exception as e:
        answer = f"Error from OpenAI:\n\n{str(e)}"

    # Show the assistant's response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Optional CSV export if report is requested
    if "report" in prompt.lower():
        df = pd.DataFrame({
            "Date": [datetime.today().strftime("%Y-%m-%d")],
            "Status": ["No anomalies"],
            "Summary": ["Normal activity levels"]
        })
        st.download_button("Download Report (CSV)", df.to_csv(index=False), "report.csv", "text/csv")

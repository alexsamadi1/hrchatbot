import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import collections
import re

def show_analytics_dashboard():
    st.title("📊 HR Chatbot Analytics")

    if not st.session_state.get("is_admin", False):
        st.error("⛔ Access denied.")
        return

    try:
        df = pd.read_csv("query_logs.csv", usecols=[
        "timestamp", "session_id", "question", "response",
        "fallback", "response_type", "user_role", "user_tenure", "source_docs", "feedback"
    ])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except FileNotFoundError:
        st.warning("No query log file found yet.")
        return
    except Exception as e:
        st.error(f"Error loading log data: {e}")
        return

    show_usage_summary(df)
    show_top_keywords(df)
    show_top_questions(df)
    show_user_demographics(df)
    show_bot_performance(df)
    show_source_documents(df)
    show_sessions(df)

    st.markdown("---")
    if st.button("🔙 Back to Assistant"):
        st.session_state.show_analytics = False
        st.rerun()


# --- Usage Summary ---
def show_usage_summary(df):
    st.subheader("📈 Usage Overview")
    st.metric("Total Queries", len(df))
    daily = df.groupby(df["timestamp"].dt.date).size()
    st.line_chart(daily.rename("Daily Queries"))


# --- Top Keywords ---
def show_top_keywords(df):
    st.subheader("🔍 Top Keywords")
    all_words = " ".join(df["question"].fillna("")).lower()
    words = re.findall(r"\b\w{4,}\b", all_words)
    common = collections.Counter(words).most_common(10)
    word_df = pd.DataFrame(common, columns=["Word", "Count"])
    st.dataframe(word_df)


# --- Most Common Questions ---
def show_top_questions(df):
    st.subheader("📌 Most Frequently Asked Questions")
    if "question" in df.columns:
        q_counts = df["question"].value_counts().head(10)
        st.table(q_counts.rename("Count"))


# --- User Demographics ---
def show_user_demographics(df):
    st.subheader("👥 User Demographics")
    col1, col2 = st.columns(2)

    if "user_role" in df.columns:
        col1.markdown("**Role**")
        col1.bar_chart(df["user_role"].value_counts())

    if "user_tenure" in df.columns:
        col2.markdown("**Tenure**")
        col2.bar_chart(df["user_tenure"].value_counts())


# --- Bot Performance ---
def show_bot_performance(df):
    st.subheader("🤖 Bot Performance")

    if "fallback" in df.columns:
        fallback_rate = df["fallback"].mean() * 100
        st.metric("Fallback Rate", f"{fallback_rate:.1f}%")

        fallback_daily = df.groupby(df["timestamp"].dt.date)["fallback"].mean() * 100
        st.line_chart(fallback_daily.rename("Fallback % Over Time"))

    if "response_type" in df.columns:
        st.markdown("**Response Type**")
        st.bar_chart(df["response_type"].value_counts())


# --- Source Docs ---
def show_source_documents(df):
    if "source_docs" not in df.columns:
        return

    st.subheader("📄 Top Source Documents")
    exploded = df["source_docs"].dropna().str.split(", ")
    flat = exploded.explode()
    doc_counts = flat.value_counts().head(10)
    st.bar_chart(doc_counts.rename("Mentions"))


# --- Sessions ---
def show_sessions(df):
    if "session_id" in df.columns:
        st.subheader("🧭 Session Analytics")
        session_count = df["session_id"].nunique()
        st.metric("Unique Sessions", session_count)

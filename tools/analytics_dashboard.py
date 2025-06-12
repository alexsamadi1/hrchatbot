import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import collections
import re
from pathlib import Path

def show_analytics_dashboard():
    st.title("📊 HR Chatbot Query Analytics")

    if not st.session_state.get("is_admin", False):
        st.error("⛔ Access denied.")
        return

    try:
        df = pd.read_csv("query_logs.csv", names=["timestamp", "question", "response"], header=None)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.metric("Total Queries Logged", len(df))

        query_counts = df.groupby(df["timestamp"].dt.date).size()
        st.line_chart(query_counts)

        all_words = " ".join(df["question"].fillna("")).lower()
        words = re.findall(r"\b\w{4,}\b", all_words)
        common_words = collections.Counter(words).most_common(10)

        st.markdown("### 🔍 Top Keywords")
        for word, count in common_words:
            st.write(f"- {word} ({count})")

    except FileNotFoundError:
        st.warning("No query log file found yet.")
    except Exception as e:
        st.error(f"Error loading log data: {e}")

    # ✅ Back to Assistant Button
    if st.button("🔙 Back to Assistant"):
        st.session_state.show_analytics = False
        st.rerun()

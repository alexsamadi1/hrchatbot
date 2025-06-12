import csv
from datetime import datetime
import os
import streamlit as st
from tools.s3_utils import download_file_from_s3, upload_file_to_s3

LOG_FILE = "query_logs.csv"
S3_BUCKET = st.secrets["S3_DOCS_BUCKET"]
S3_KEY = f"logs/{LOG_FILE}"  # <- Keeps log files separated in the bucket

def ensure_log_file_exists():
    """Check if the log file exists locally. If not, download from S3."""
    if not os.path.exists(LOG_FILE):
        try:
            download_file_from_s3(S3_KEY, S3_BUCKET)
            print(f"[LOG] Pulled {LOG_FILE} from S3")
        except Exception as e:
            print(f"[LOG] No existing log on S3 or error downloading: {e}")

def log_query_to_csv(user_input: str, response: str):
    """Append a query and response to the log file and upload to S3."""
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([datetime.now().isoformat(), user_input.strip(), response.strip()])
    try:
        upload_file_to_s3(LOG_FILE, S3_KEY, S3_BUCKET)
        print(f"[LOG] Uploaded updated log to S3.")
    except Exception as e:
        print(f"[LOG] Failed to upload log to S3: {e}")
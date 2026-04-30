'''
Brand Guardian - Streamlit Frontend
Calls the FastAPI /audit endpoint and displays the compliance report.
'''

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

# Page config
st.set_page_config(
    page_title="Brand Guardian",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Brand Guardian")
st.subheader("AI-Powered Video Compliance Auditor")
st.markdown("Submit a YouTube URL to automatically audit the video against brand compliance rules.")

st.divider()

# Input
video_url = st.text_input(
    label="YouTube URL",
    placeholder="https://youtu.be/...",
)

run_audit = st.button("Run Audit", type="primary", use_container_width=True)

if run_audit:
    if not video_url.strip():
        st.warning("Please enter a YouTube URL.")
    else:
        with st.spinner("Running compliance audit... this may take a few minutes."):
            try:
                response = requests.post(
                    f"{API_URL}/audit",
                    json={"video_url": video_url},
                    timeout=600
                )

                if response.status_code == 200:
                    data = response.json()

                    st.divider()

                    # Status banner
                    status = data.get("status", "UNKNOWN")
                    if status == "PASS":
                        st.success(f"Result: {status} — No violations detected.")
                    else:
                        st.error(f"Result: {status} — Violations detected.")

                    # Metadata
                    st.caption(f"Session ID: {data.get('session_id')} | Video ID: {data.get('video_id')}")

                    # Violations
                    violations = data.get("compliance_results", [])
                    if violations:
                        st.markdown("### Violations")
                        for issue in violations:
                            severity = issue.get("severity", "")
                            category = issue.get("category", "")
                            description = issue.get("description", "")

                            if severity == "CRITICAL":
                                st.error(f"**[{severity}] {category}**\n\n{description}")
                            else:
                                st.warning(f"**[{severity}] {category}**\n\n{description}")

                    # Final report
                    st.markdown("### Summary Report")
                    st.info(data.get("final_report", "No report generated."))

                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Make sure the FastAPI server is running.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The video may be too long to process.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

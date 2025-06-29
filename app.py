import os
import markdown2
import streamlit as st
from utils.resume_parser import parse_resume
from utils.score_resume import score_resume
from utils.gemini_helper import gemini_resume_helper

# ---------------------- Streamlit UI ----------------------
extra,maincol, extra2 = st.columns([1, 5, 1])
st.markdown(
    """
    <div style='text-align: center'>
        <h1 style='font-size: 100px;'>ResumAI</h1>
        <h3>AI Resume reader and recommender</h3>
        <p>Upload your resume, and this tool will match it with your job description and recommend ways to improve it.</p>
        <h4>🔧 How it works:</h4>
        <ul style="list-style: none; padding-left: 0;">
            <li>📄 Upload a <strong>.pdf</strong>, <strong>.txt</strong>, or <strong>.docx</strong> resume</li>
            <li>📝 Paste a job description</li>
            <li>✅ Get a match score and keyword feedback</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------- Session State ----------------------
if "show_feedback_dialog" not in st.session_state:
    st.session_state["show_feedback_dialog"] = False

if "ai_response" not in st.session_state:
    st.session_state["ai_response"] = ""
    
if not os.path.exists("assets"):
    os.makedirs("assets")

# ---------------------- Dialog Definition ----------------------
@st.dialog("AI Resume Feedback", width="large")
def show_ai_feedback():
    st.markdown(st.session_state["ai_response"], unsafe_allow_html=True)
    
@st.dialog("Score", width="large")
def show_score_and_missing_words(result):
    st.success(f"Your resume matches the job description with a score of {result['score']:.2f}%.")
    st.markdown("**Missing keywords:**")
    st.write(", ".join(sorted(result["missing"])))
    
#------------ Simplifying functions ------------------------
def get_score(job_description, uploaded_file) -> dict:
    with st.spinner("Running your resume through our system..."):
        result = score_resume(uploaded_file, job_description)
        if "error" in result:
            st.error(result["error"])
        else:
            show_score_and_missing_words(result)
            st.session_state["resume_text"] = result["resume_text"]
            st.session_state["job_text"] = result["job_text"]
    return result

# ---------------------- Inputs ----------------------
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt", "docx"])
job_description = st.text_area("Paste your job description here", max_chars=999999)


# --------------Buttons functionality -------------------------
col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 2, 3])
with col2:
    if st.button("Show score"):
        if job_description and uploaded_file:
            st.session_state["has_file"] = True
            result = get_score(job_description, uploaded_file)
        else:
            st.session_state["has_file"] = False
            

with col4:
    if st.button("Improve Resume"):
        if job_description and uploaded_file:
            st.session_state["has_file"] = True
            with st.spinner("Improving your resume..."):
                ai_md = gemini_resume_helper(
                    st.session_state["resume_text"],
                    st.session_state["job_text"]
                )
                html_response = markdown2.markdown(ai_md)
                st.session_state["ai_response"] = html_response
                st.session_state["show_feedback_dialog"] = True
        else:
            st.session_state["has_file"] = False

if st.session_state.get("has_file") is False:
    st.error("Please fill in the job description and upload your resume!")
    

# ---------------------- Show Dialog ----------------------
if st.session_state.get("show_feedback_dialog", False):
    show_ai_feedback()
    st.session_state["show_feedback_dialog"] = False

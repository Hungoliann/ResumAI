from utils.resume_parser import parse_resume
from utils.score_resume import score_resume
from utils.gemini_helper import gemini_resume_helper
import streamlit as st
import os
import markdown 

# ---------------------- Streamlit UI ----------------------
st.title("ResumAI")
st.header("AI Resume reader and recommender")

st.write("This tool lets you upload a resume file, extracts the content, and compares it with a job description you provide.")
st.markdown("### 🔧 How it works:")
st.markdown("""
- Upload a `.pdf` or `.txt` resume  
- Paste a job description  
- Get a match score and keyword feedback
""")

# ---------------------- Session State ----------------------
if "show_feedback_modal" not in st.session_state:
    st.session_state["show_feedback_modal"] = False

if "ai_response" not in st.session_state:
    st.session_state["ai_response"] = ""

if not os.path.exists("assets"):
    os.makedirs("assets")

# ---------------------- Inputs ----------------------
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt", "docx"])
job_description = st.text_area("Paste your job description here", max_chars=999999)

col1, col2 = st.columns(2)

# ---------------------- Score Button ----------------------
with col1:
    if st.button("Show score"):
        if job_description and uploaded_file:
            with st.spinner("Running your resume through our system..."):
                result = score_resume(uploaded_file, job_description)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Your resume matches the job description with a score of {result['score']:.2f}%.")
                    if result["missing"]:
                        st.markdown("**Missing keywords:**")
                        st.write(", ".join(sorted(result["missing"])))

                    st.session_state["resume_text"] = result["resume_text"]
                    st.session_state["job_text"] = result["job_text"]
                    st.session_state["show_improve_button"] = True
        else:
            st.error("Please fill in the job description and upload your resume!")

# ---------------------- Improve Button ----------------------
with col2:
    if st.session_state.get("show_improve_button", False):
        if st.button("Click to improve your resume"):
            with st.spinner("Improving your resume..."):
                ai_md = gemini_resume_helper(
                    st.session_state["resume_text"],
                    st.session_state["job_text"]
                )
                html_response = markdown.markdown(ai_md)
                st.session_state["ai_response"] = html_response
                st.session_state["show_feedback_modal"] = True

# ---------------------- Optional: Blur Background ----------------------
if st.session_state.get("show_feedback_modal", False):
    st.markdown("""
    <style>
    .blur-container {
        filter: blur(4px);
        pointer-events: none;
        user-select: none;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="blur-container">', unsafe_allow_html=True)

# All UI content goes here (would normally be blurred)
# For now, nothing more to add here

if st.session_state.get("show_feedback_modal", False):
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Modal ----------------------
if st.session_state.get("show_feedback_modal", False):
    st.markdown(f"""
    <style>
    .modal {{
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(0, 0, 0, 0.6);
        display: flex; align-items: center; justify-content: center;
        z-index: 9999;
    }}
    .modal-content {{
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        width: 80%;
        max-width: 600px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        position: relative;
        overflow-y: auto;
        max-height: 80vh;
    }}
    .close-button {{
        position: absolute;
        top: 10px; right: 20px;
        font-size: 24px;
        cursor: pointer;
    }}
    </style>

    <div class="modal">
        <div class="modal-content">
            <span class="close-button" onclick="window.location.reload()">×</span>
            <h3>AI Resume Feedback</h3>
            <div>{st.session_state['ai_response']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

from utils.resume_parser import parse_resume
from utils.match_resume_to_job import match_resume_to_job
from utils.gemini_helper import gemini_resume_helper
import streamlit as st
import os
import time



st.title("ResumAI")
st.header("AI Resume reader and recommender")


st.write("This tool lets you upload a resume file, extracts the content, and compares it with a job description you provide.")
st.markdown("### 🔧 How it works:")
st.markdown("""
- Upload a `.pdf` or `.txt` resume  
- Paste a job description  
- Get a match score and keyword feedback
""")
if not os.path.exists("assets"): 
    os.makedirs("assets")
    
uploaded_file = st.file_uploader("Upload your resume", type= ["pdf", "txt", "docx"])

job_description = st.text_area("Paste your job description here", max_chars = 999999)


if st.button("Submit"):
    if job_description and uploaded_file:
        with st.spinner("Running your resume through our system..."):
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext not in [".pdf", ".txt", ".docx"]:
                st.error("Unsupported file type! Please upload a PDF, TXT, or DOCX file.")
            else:
                temp_path = os.path.join("assets", f"temp_resume{ext}")
                
                with open(temp_path, "wb") as resume_file:
                    resume_file.write(uploaded_file.getbuffer())

                parsed_text = parse_resume(temp_path, ext)
                
                if parsed_text and job_description.strip():

                    score, missing = match_resume_to_job(parsed_text, job_description)
                    # Display result
                    st.success(f"Your resume matches the job description with a score of {score:.2f}%.")
                    if missing:
                        st.markdown("**Missing keywords:**")
                        st.write(",  ".join(sorted(missing))) 
                    # stores data across the submit interaction and set flag in order for next button to show
                    st.session_state["resume_text"] = parsed_text
                    st.session_state["job_text"] = job_description
                    st.session_state["show_improve_button"] = True
    else:
        st.error("Please fill in the job description and u pload your resume!")

####### SHOW IMPROVE BUTTON IF SCORE IS DISPLAYED ##############################
if st.session_state.get("show_improve_button", False):
    if st.button("Click to improve your resume"):
        with st.spinner("Improving your resume"):
            ai_reponse = gemini_resume_helper(st.session_state["resume_text"], 
                                              st.session_state["job_text"])
            st.write(ai_reponse)   



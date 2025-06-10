from utils.resume_parser import parse_resume
import os
import streamlit as st

st.title("ResumAI")
st.header("AI Resume reader and recommender")
st.subheader("Rate your resume and edit it based on your job description")


st.write("This tool lets you upload a resume file, extracts the content, and compares it with a job description you provide.")
st.markdown("### 🔧 How it works:")
st.markdown("""
- Upload a `.pdf` or `.txt` resume  
- Paste a job description  
- Get a match score and keyword feedback
""")

uploaded_file = st.file_uploader("Upload your resume", type= ["pdf", "txt", "docx"])
with st.spinner("Running your resume through our system..."):
    while uploaded_file: 
        # Step: Save it temporarily so parse_resume() can read it
        if os.path.exists("assets"):
            with open(r"assets\temp_resume", "wb") as resume:
                # get bytes from the streamlit upload
                resume.write(uploaded_file.getbuffer())
                parsed_text = parse_resume(resume)
                break
        else:
            os.makedirs("ResumAI/assets")
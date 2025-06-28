import os
from utils.resume_parser import parse_resume
from utils.match_resume_to_job import match_resume_to_job

def score_resume( uploaded_file,job_description,):
    if not uploaded_file:
        return{"error" : "No file uploaded"}
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in [".pdf", ".txt", ".docx"]:
        return {"error":"Unsupported file type! Please upload a PDF, TXT, or DOCX file."}
    else:
        temp_path = os.path.join("assets", f"temp_resume{ext}")
        
        with open(temp_path, "wb") as resume_file:
            resume_file.write(uploaded_file.getbuffer())

        parsed_text = parse_resume(temp_path, ext)
        
        if parsed_text and job_description.strip():

            score, missing = match_resume_to_job(parsed_text, job_description)
            return {
                "score" : score,
                "missing" : missing,
                "resume_text" : parsed_text,
                "job_text": job_description
            }
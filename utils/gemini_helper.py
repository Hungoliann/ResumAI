from google import genai
import os
from google.genai import types
import streamlit as st 



def gemini_resume_helper(resume_text:str, job_text: str) -> str:
    api_keys = st.secrets["api_key"]
    client = genai.Client(api_key= api_keys)
    chat = client.chats.create(model = 'models/gemini-1.5-flash')
   ### if not user_input:
    response = chat.send_message(f"""You are a helpful and experienced technical recruiter and resume advisor.
                Below is a job description and a resume. Your task is to review both, and provide helpful, actionable feedback to improve the resume so it better matches the job description.
                Focus on:
                - Missing skills or technologies
                - Suggestions for rewording or adding stronger action verbs
                - Specific improvements that make the resume more relevant to the job
                - Any formatting or clarity issues
                --- Job Description ---
                {job_text}
                    Resume Text ---
                {resume_text}
                Please output your response in a clear and structured way using markdown formatting with headers like:
                1. Summary Feedback
                2. Suggested Resume Edits
                3. Missing Keywords or Technologies 
            """)
    ###else: 
        ####reponse = chat.send_message(user_input)
    return response.text

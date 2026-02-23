
from google import genai
import streamlit as st


def gemini_resume_helper(resume_text: str, job_text: str) -> str:
    """
    Uses Gemini to provide actionable resume feedback based on a job description.
    Returns markdown-formatted suggestions.
    """
    client = genai.Client(api_key=st.secrets["api_key"])
    chat = client.chats.create(model='models/gemini-1.5-flash')

    response = chat.send_message(f"""You are a helpful and experienced technical recruiter and resume advisor.
        Below is a job description and a resume. Your task is to review both and provide
        helpful, actionable feedback to improve the resume so it better matches the job description.

        Focus on:
        - Missing skills or technologies
        - Suggestions for rewording or adding stronger action verbs
        - Specific improvements that make the resume more relevant to the job
        - Any formatting or clarity issues

        --- Job Description ---
        {job_text}

        --- Resume Text ---
        {resume_text}

        Please output your response in a clear and structured way using markdown formatting:
        1. Summary Feedback
        2. Suggested Resume Edits
        3. Missing Keywords or Technologies
    """)

    return response.text
import re
    
    
"""
Compare resume text with job description text and return
a matching score (%) and list of missing keywords.

Args:
    resume_text (str): Text extracted from the resume.
    job_desc_text (str): Text of the job description.

Returns:
    tuple: (match_score as float, list of missing keywords)
"""

def match_resume_to_job(resume_text: str, job_desc_text: str) -> tuple[float, list[str]]:

    # Remove punctuation and convert to lowercase for both texts
    resume_text = re.sub(r"[^\w\s]", "", resume_text).lower()
    job_desc_text = re.sub(r"[^\w\s]", "", job_desc_text).lower()

    # Tokenize the texts by splitting on whitespace
    resume_tokens = resume_text.split()
    job_tokens = job_desc_text.split()

    # Convert token lists to sets for faster membership testing and unique words
    resume_set = set(resume_tokens)
    job_set = set(job_tokens)

    # Find common words between resume and job description
    common_words = resume_set.intersection(job_set)

    # Calculate matching score as percentage of job description keywords found in resume
    score = (len(common_words) / len(job_set)) * 100 if job_set else 0

    # Identify which job description keywords are missing in the resume
    missing_words = job_set - resume_set

    return score, list(missing_words)

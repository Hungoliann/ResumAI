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
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
 "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def match_resume_to_job(resume_text: str, job_desc_text: str) -> tuple[float, list[str]]:

    # Remove punctuation and convert to lowercase for both texts
    resume_text = re.sub(r"[^\w\s]", "", resume_text).lower()
    job_desc_text = re.sub(r"[^\w\s]", "", job_desc_text).lower()

    # Tokenize the texts by splitting on whitespace
    resume_tokens = resume_text.split()
    job_tokens = job_desc_text.split()

    # Filter stopwords from texts 
    resume_filtered = [w for w in resume_tokens if w not in stop_words]
    job_filtered = [w for w in job_tokens if w not in stop_words]

    # Convert token lists to sets for faster membership testing and unique words
    resume_set = set(resume_filtered)
    job_set = set(job_filtered)

    # Find common words between resume and job description
    common_words = resume_set.intersection(job_set)

    # Calculate matching score as percentage of job description keywords found in resume
    score = (len(common_words) / len(job_set)) * 100 if job_set else 0

    # Identify which job description keywords are missing in the resume
    missing_words = job_set - resume_set

    return score, list(missing_words)

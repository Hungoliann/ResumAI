import os
from utils.resume_parser import ResumeParser
from utils.match_resume_to_job import JobDescription, Resume, ResumeMatcher


class ScoreResume:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

    def __init__(self, uploaded_file, job_description: str):
        self.file = uploaded_file
        self.job = job_description
        self.parsed_text = ""
        self.ext = os.path.splitext(uploaded_file.name)[1].lower()
        self.missing = []
        self.result = {}
        self.score = None
        self.temp_path = None

    def validate(self):
        if not self.file:
            self.result = {"error": "No file uploaded"}
        elif self.ext not in self.SUPPORTED_EXTENSIONS:
            self.result = {"error": f"Unsupported file type '{self.ext}'. Please upload a PDF, TXT, or DOCX file."}
        return self

    def save(self):
        if "error" in self.result:
            return self
        self.temp_path = os.path.join("assets", f"temp_resume{self.ext}")
        with open(self.temp_path, "wb") as f:
            f.write(self.file.getbuffer())
        return self

    def get_parse(self):
        if "error" in self.result:
            return self
        self.parsed_text = ResumeParser(self.temp_path).parser()
        return self

    def score_resume(self):
        if "error" in self.result:
            return self
        if not self.parsed_text or not self.job:
            self.result = {"error": "Missing resume content or job description"}
            return self

        stop_words = set(ResumeMatcher._default_stopwords())
        matcher = ResumeMatcher(self.parsed_text, self.job)
        resume_obj = Resume(self.parsed_text, stop_words)
        job_obj = JobDescription(self.job, stop_words)
        self.score, self.missing = matcher.compute_similarity(resume_obj, job_obj)
        return self

    def get_result(self) -> dict:
        if "error" in self.result:
            return self.result
        return {
            "score": self.score,
            "missing": self.missing,
            "resume_text": self.parsed_text,
            "job_text": self.job
        }
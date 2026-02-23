# ResumAI

An AI-powered web app that analyzes your resume against a job description to help you improve your chances of getting hired.

## Features

- **Match Score** — Semantic similarity score between your resume and job description
- **Neural Score** — Score from a fine-tuned neural ranker trained on resume-job pairs
- **Missing Keywords** — Terms in the job description not found in your resume
- **AI Feedback** — Structured suggestions from Gemini AI for improving your resume
- **Bullet Analyzer** — Flags weak bullet points by relevance and action verb strength
- **Multi-format** — Supports PDF, DOCX, and TXT resumes

## Setup

```bash
git clone https://github.com/Hungoliann/ResumAI.git
cd ResumAI
pip install -r requirements.txt
```

Add your Gemini API key to `.streamlit/secrets.toml`:

```toml
api_key = "your-gemini-api-key-here"
```

Optionally fine-tune the neural ranker:

```bash
python utils/resume_ranker_nn.py
```

Run the app:

```bash
streamlit run app.py
```

## Tech Stack

- **UI** — Streamlit
- **Embeddings & Neural Ranker** — `sentence-transformers`, PyTorch
- **AI Feedback** — Google Gemini 1.5 Flash
- **PDF parsing** — PyMuPDF
- **DOCX parsing** — python-docx


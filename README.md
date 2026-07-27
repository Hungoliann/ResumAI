# ResumAI

A web app that scores your resume against a specific job description and tells you what to fix before you apply.

## The problem

Most applications get filtered before a human reads them. Resumes are ranked against the posting, but applicants never see that ranking or learn which parts of the posting they failed to address. ResumAI makes the gap visible and specific.

## Features

- **Match score.** Semantic similarity between your resume and the job description
- **Neural score.** A separate score from a neural ranker fine-tuned on resume and job pairs
- **Missing keywords.** Terms the posting asks for that your resume never mentions
- **AI feedback.** Structured rewrite suggestions from Gemini 1.5 Flash
- **Bullet analyzer.** Flags weak bullets by relevance and action verb strength
- **Multi-format input.** PDF, DOCX, and TXT


## How it works

The app produces two independent scores rather than one, because they fail in different ways.

**Semantic similarity.** Resume text and job description are embedded with `sentence-transformers` and compared by cosine similarity. This is fast and needs no training data, but it rewards surface overlap and can score a keyword-stuffed resume highly.

**Neural ranker.** A small PyTorch model fine-tuned on labeled resume and job pairs, which learns from examples of what actually counts as a good match rather than from raw text overlap. Training script is `utils/resume_ranker_nn.py`.

Keyword gaps are extracted separately by pulling skill and requirement terms from the posting and checking them against the parsed resume text. The bullet analyzer scores each bullet on relevance to the posting and on action verb strength, then surfaces the weakest ones. Gemini 1.5 Flash takes the resume, the posting, and the gap list and returns targeted rewrite suggestions.


## Stack

Python, Streamlit, PyTorch, sentence-transformers, Google Gemini 1.5 Flash, PyMuPDF, python-docx

## Setup

```
git clone https://github.com/Hungoliann/ResumAI.git
cd ResumAI
pip install -r requirements.txt
```

Add your Gemini API key to `.streamlit/secrets.toml`:

```
api_key = "your-gemini-api-key-here"
```

Run the app:

```
streamlit run app.py
```

To retrain the neural ranker:

```
python utils/resume_ranker_nn.py
```

## Project structure

```
app.py                        Streamlit UI and scoring orchestration
utils/resume_ranker_nn.py     Neural ranker training and inference
utils/                        Parsing, keyword extraction, bullet analysis
.streamlit/                   Config and secrets
```

## Status

Working: PDF, DOCX, and TXT parsing, both scoring paths, keyword gap detection, bullet analysis, Gemini feedback.

Known limitations: Small training set

Next: Revamped UI experience on Streamlit

## Why I built it

I was applying to ML and software internships and wanted a faster way to decide whether a posting was worth tailoring for. The similarity score was straightforward. Getting feedback that was actually specific enough to act on was the harder part, and the neural ranker exists because plain cosine similarity kept rating keyword-heavy resumes above better ones.

## License

MIT

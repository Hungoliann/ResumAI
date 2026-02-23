"""
resume_ranker_nn.py

Fine-tunes a SentenceTransformer encoder using a contrastive loss (CosineSimilarityLoss)
so that resume-job pairs learn a task-specific embedding space — rather than relying on
general-purpose embeddings from a frozen model.

Why this matters over cosine similarity on frozen embeddings:
- Frozen models encode semantic similarity in general text, not job-resume relevance specifically
- Fine-tuning pulls matching pairs closer and pushes mismatches apart in embedding space
- The learned representations capture domain-specific relevance signals (e.g. "Python" in a
  data science JD matters more than "Python" in a herpetology JD)

Training uses CosineSimilarityLoss with soft labels, so the model learns a
continuous relevance scale rather than hard match/no-match classification.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_training_examples() -> tuple[list[InputExample], list[InputExample]]:
    """
    Build synthetic resume-job training pairs with soft similarity labels.

    Labels:
      1.0  = strong match (skills + domain align well)
      0.6-0.8 = partial match (related field, some skill overlap)
      0.0-0.2 = mismatch (unrelated field or skills)

    In production you'd replace this with real labeled pairs from recruiter
    feedback, click-through data, or hiring outcomes.
    """
    train_pairs = [
        InputExample(texts=[
            "Python developer with 3 years experience in NLP, transformers, and PyTorch. "
            "Built text classification pipelines and fine-tuned BERT for sentiment analysis.",
            "Seeking ML engineer with Python and NLP expertise. Experience with transformer "
            "models and deep learning frameworks like PyTorch or TensorFlow required."
        ], label=0.95),

        InputExample(texts=[
            "Data scientist skilled in pandas, scikit-learn, SQL, and statistical modeling. "
            "Deployed ML models to production using Flask and Docker.",
            "Data scientist role requiring Python, SQL, statistical analysis, and experience "
            "deploying machine learning models in a production environment."
        ], label=0.92),

        InputExample(texts=[
            "Full-stack engineer with React, Node.js, and PostgreSQL. Built REST APIs and "
            "single-page applications. Experience with CI/CD pipelines and AWS.",
            "Full-stack developer needed with React frontend skills, Node.js backend, "
            "PostgreSQL database experience, and cloud deployment knowledge."
        ], label=0.93),

        InputExample(texts=[
            "Computer vision researcher with OpenCV and PyTorch. Published work on "
            "object detection and image segmentation using YOLO and Mask R-CNN.",
            "Computer vision engineer role. Requires PyTorch, experience with object "
            "detection models, and knowledge of image processing techniques."
        ], label=0.90),

        InputExample(texts=[
            "Software engineer with Java and Spring Boot. Designed microservices and "
            "REST APIs. Strong background in system design and distributed systems.",
            "Backend Java engineer. Must have Spring Boot experience, REST API design, "
            "and understanding of microservices architecture."
        ], label=0.91),

        InputExample(texts=[
            "Python developer with data analysis experience using pandas and matplotlib. "
            "Some exposure to scikit-learn for regression tasks.",
            "Seeking ML engineer with Python and NLP expertise. Experience with transformer "
            "models and deep learning frameworks like PyTorch or TensorFlow required."
        ], label=0.55),

        InputExample(texts=[
            "Data analyst with Excel, Tableau, and SQL. Built dashboards and reported KPIs. "
            "Basic Python scripting for data cleaning.",
            "Data scientist role requiring Python, SQL, statistical analysis, and experience "
            "deploying machine learning models in a production environment."
        ], label=0.45),

        InputExample(texts=[
            "Frontend developer with HTML, CSS, and Vue.js. Some experience with REST APIs "
            "and basic Node.js for local development.",
            "Full-stack developer needed with React frontend skills, Node.js backend, "
            "PostgreSQL database experience, and cloud deployment knowledge."
        ], label=0.50),

        InputExample(texts=[
            "Machine learning engineer with experience in tabular data and gradient boosting. "
            "Familiar with CNNs from coursework but no production CV experience.",
            "Computer vision engineer role. Requires PyTorch, experience with object "
            "detection models, and knowledge of image processing techniques."
        ], label=0.40),

        InputExample(texts=[
            "Python developer with Django and PostgreSQL. Deployed web apps but limited "
            "experience with distributed systems or microservices.",
            "Backend Java engineer. Must have Spring Boot experience, REST API design, "
            "and understanding of microservices architecture."
        ], label=0.35),

        InputExample(texts=[
            "Graphic designer with Photoshop, Illustrator, and Figma. Created brand "
            "identities and marketing materials for e-commerce clients.",
            "Seeking ML engineer with Python and NLP expertise. Experience with transformer "
            "models and deep learning frameworks like PyTorch or TensorFlow required."
        ], label=0.05),

        InputExample(texts=[
            "High school biology teacher with 5 years classroom experience. Curriculum "
            "development and student assessment. No programming background.",
            "Data scientist role requiring Python, SQL, statistical analysis, and experience "
            "deploying machine learning models in a production environment."
        ], label=0.05),

        InputExample(texts=[
            "Marketing manager with SEO, Google Ads, and content strategy. Managed "
            "campaigns with $500K budgets and grew organic traffic by 40%.",
            "Full-stack developer needed with React frontend skills, Node.js backend, "
            "PostgreSQL database experience, and cloud deployment knowledge."
        ], label=0.02),

        InputExample(texts=[
            "Accountant with CPA certification, tax preparation, and financial reporting. "
            "QuickBooks and Excel proficiency.",
            "Computer vision engineer role. Requires PyTorch, experience with object "
            "detection models, and knowledge of image processing techniques."
        ], label=0.02),

        InputExample(texts=[
            "Registered nurse with ICU experience, patient care, and medical charting. "
            "Strong communication and clinical decision-making skills.",
            "Backend Java engineer. Must have Spring Boot experience, REST API design, "
            "and understanding of microservices architecture."
        ], label=0.02),
    ]

    val_pairs = [
        InputExample(texts=[
            "NLP engineer with Hugging Face, BERT fine-tuning, and text classification. "
            "Experience building chatbots and question-answering systems.",
            "Seeking ML engineer with Python and NLP expertise. Experience with transformer "
            "models required."
        ], label=0.90),

        InputExample(texts=[
            "iOS developer with Swift and Xcode. Published 3 apps on the App Store.",
            "Seeking ML engineer with Python and NLP expertise. Experience with transformer "
            "models required."
        ], label=0.08),
    ]

    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

class ResumeRankerFineTuner:
    """
    Fine-tunes a SentenceTransformer on resume-job similarity pairs using
    CosineSimilarityLoss. The encoder weights are updated end-to-end so the
    model learns a relevance-specific embedding space.
    """

    def __init__(self, base_model: str = 'all-MiniLM-L6-v2', output_dir: str = 'models/resume_ranker'):
        self.base_model = base_model
        self.output_dir = output_dir
        self.model = SentenceTransformer(base_model)

    def train(
        self,
        train_examples: list[InputExample],
        val_examples: list[InputExample],
        epochs: int = 10,
        batch_size: int = 8,
        warmup_ratio: float = 0.1,
    ):
        """
        Fine-tune with CosineSimilarityLoss and evaluate on validation set each epoch.

        CosineSimilarityLoss minimizes MSE between predicted cosine similarity and
        the soft label, directly optimizing embedding geometry for this task.
        """
        loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        loss_fn = losses.CosineSimilarityLoss(self.model)

        # Evaluator computes Pearson/Spearman correlation on val set
        val_resumes = [ex.texts[0] for ex in val_examples]
        val_jobs    = [ex.texts[1] for ex in val_examples]
        val_labels  = [ex.label for ex in val_examples]
        evaluator = EmbeddingSimilarityEvaluator(val_resumes, val_jobs, val_labels)

        warmup_steps = int(len(loader) * epochs * warmup_ratio)

        print(f"Fine-tuning '{self.base_model}' for {epochs} epochs...")
        self.model.fit(
            train_objectives=[(loader, loss_fn)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=self.output_dir,
            show_progress_bar=True,
            save_best_model=True,     
        )
        print(f"Best model saved to: {self.output_dir}")

    def load_best(self):
        """Load the best checkpoint saved during training."""
        self.model = SentenceTransformer(self.output_dir)
        print(f"Loaded fine-tuned model from '{self.output_dir}'")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class ResumeRanker:
    """
    Inference wrapper for the fine-tuned model.
    Scores a resume against a job description, returning a float in [0, 100].
    Can also rank a list of resumes for a given job.
    """

    def __init__(self, model_path: str = 'models/resume_ranker'):
        if os.path.exists(model_path):
            self.model = SentenceTransformer(model_path)
            print(f"Using fine-tuned model from '{model_path}'")
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("No fine-tuned model found — using base model.")

    def score(self, resume_text: str, job_text: str) -> float:
        """Return relevance score in [0, 100]."""
        embs = self.model.encode([resume_text, job_text], convert_to_tensor=True).cpu().numpy()
        sim = cosine_similarity([embs[0]], [embs[1]])[0][0]
        return round(float(sim) * 100, 2)

    def rank(self, resumes: list[str], job_text: str) -> list[dict]:
        """
        Rank a list of resumes for a job description.
        Returns list of dicts sorted by score descending.
        """
        job_emb = self.model.encode(job_text, convert_to_tensor=True).cpu().numpy()
        resume_embs = self.model.encode(resumes, convert_to_tensor=True).cpu().numpy()

        scores = cosine_similarity([job_emb], resume_embs)[0]

        ranked = sorted(
            [{"resume": r, "score": round(float(s) * 100, 2)} for r, s in zip(resumes, scores)],
            key=lambda x: x["score"],
            reverse=True
        )
        return ranked


# ---------------------------------------------------------------------------
# Evaluation: before vs after fine-tuning
# ---------------------------------------------------------------------------

def evaluate_improvement(train_examples, val_examples):
    """
    Compare base model vs fine-tuned model on val set using Spearman correlation.
    Prints a side-by-side score table to show the impact of fine-tuning.
    """
    base = SentenceTransformer('all-MiniLM-L6-v2')

    print("\n--- Before vs After Fine-Tuning ---")
    print(f"{'Resume snippet':<45} {'Label':>6} {'Base':>6} {'Tuned':>6}")
    print("-" * 68)

    tuned_ranker = ResumeRanker('models/resume_ranker')

    for ex in val_examples:
        snippet = ex.texts[0][:42] + "..."
        label = ex.label

        base_embs = base.encode(ex.texts, convert_to_tensor=True).cpu().numpy()
        base_score = round(float(cosine_similarity([base_embs[0]], [base_embs[1]])[0][0]) * 100, 2)
        tuned_score = tuned_ranker.score(ex.texts[0], ex.texts[1])

        print(f"{snippet:<45} {label*100:>5.1f}% {base_score:>5.1f}% {tuned_score:>5.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_examples, val_examples = build_training_examples()

    tuner = ResumeRankerFineTuner()
    tuner.train(train_examples, val_examples, epochs=10, batch_size=4)
    tuner.load_best()

    evaluate_improvement(train_examples, val_examples)

    print("\n--- Ranking demo ---")
    job = "Seeking ML engineer with Python, NLP, and transformer model experience."
    candidates = [
        "NLP researcher with Hugging Face and BERT fine-tuning experience.",
        "Frontend developer with React and CSS, no ML background.",
        "Data scientist with Python, scikit-learn, and some NLP project work.",
    ]

    ranker = ResumeRanker('models/resume_ranker')
    for i, result in enumerate(ranker.rank(candidates, job), 1):
        print(f"#{i} ({result['score']}/100): {result['resume'][:60]}")
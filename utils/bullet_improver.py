import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class BulletImprover:
    """
    Scores and ranks resume bullet points by semantic relevance to a job description.
    Identifies weak bullets so the user knows which ones to prioritize rewriting.
    """

    WEAK_VERBS = {
        "worked", "helped", "assisted", "did", "made", "was responsible for",
        "responsible for", "involved in", "participated in", "supported"
    }

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def extract_bullets(self, resume_text: str) -> list[str]:
        """Extract bullet-like sentences from resume text."""
        lines = resume_text.split("\n")
        bullets = []
        for line in lines:
            line = line.strip()
            # Keep lines that look like bullet points or action-oriented sentences
            if len(line) > 20 and re.match(r'^[-•*]?\s*[A-Z]', line):
                bullets.append(re.sub(r'^[-•*]\s*', '', line))
        return bullets

    def score_bullets(self, bullets: list[str], job_text: str) -> list[dict]:
        """
        Score each bullet point by cosine similarity to the job description embedding.
        Returns bullets sorted from least to most relevant.
        """
        if not bullets:
            return []

        job_emb = self.model.encode(job_text, convert_to_tensor=True).cpu().numpy()
        bullet_embs = self.model.encode(bullets, convert_to_tensor=True).cpu().numpy()

        scores = cosine_similarity([job_emb], bullet_embs)[0]

        results = [
            {
                "bullet": bullet,
                "score": round(float(score) * 100, 2),
                "weak": self._is_weak(bullet)
            }
            for bullet, score in zip(bullets, scores)
        ]

        # Sort ascending so weakest bullets appear first
        return sorted(results, key=lambda x: x["score"])

    def _is_weak(self, bullet: str) -> bool:
        """Flag bullets that start with weak or vague action verbs."""
        first_words = bullet.lower().strip()
        return any(first_words.startswith(verb) for verb in self.WEAK_VERBS)

    def get_weakest(self, bullets: list[str], job_text: str, n: int = 3) -> list[dict]:
        """Return the n lowest-scoring bullets for prioritized rewriting."""
        scored = self.score_bullets(bullets, job_text)
        return scored[:n]
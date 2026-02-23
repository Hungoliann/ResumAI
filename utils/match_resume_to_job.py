import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
    
    

class Resume:
    def __init__(self, text: str, stop_words : set):
        self.raw = text
        self.cleaned_token = self._clean_and_tokenized(self.raw, stop_words)
        self.embedding = None
    
    def _clean_and_tokenized(self, text, stop_words):
        text = re.sub(r"[^\w\s]", "", text).lower()
        return [word for word in text.split() if word not in stop_words]
    
    def compute_embedding(self, model):
        if self.embedding is None: 
            joined = " ".join(self.cleaned_token)
            self.embedding = model.encode(joined, convert_to_tensor=True)
        return self.embedding
        
class JobDescription:
    def __init__(self, desc: str, stop_words: set):
        self.raw = desc
        self.cleaned_desc = self._clean_and_tokenized(desc, stop_words)
        self.embedding = None
    
    def _clean_and_tokenized(self, desc, stop_words):
        desc = re.sub(r"[^\w\s]", "", desc).lower()
        return [word for word in desc.split() if word not in stop_words]
    
    def compute_embedding(self, model):
        if self.embedding is None:
            joined = " ".join(self.cleaned_desc)
            self.embedding = model.encode(joined, convert_to_tensor=True)
        return self.embedding

class ResumeMatcher:
    @staticmethod
    def _default_stopwords():
        return [
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
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
    
    def __init__ (self, parsed_text, job_desc):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text = parsed_text
        self.job = job_desc

    def compute_similarity(self, resume: Resume, job: JobDescription) -> tuple[float, list[str]]:
        resume_emb = resume.compute_embedding(self.model).cpu().numpy()        
        job_emb = job.compute_embedding(self.model)
        
        similarity = cosine_similarity([job_emb], [resume_emb])[0][0]
        score = round((similarity) *100, 2)
        
        missing = set(job.cleaned_desc) - set(resume.cleaned_token)
        
        return score, list(missing)

        

        

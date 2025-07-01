import pymupdf
import re
import os
from docx import Document

class ResumeParser:
    def __init__(self,file_path:str):
        self.file_path = file_path
        self.extension = os.path.splitext(file_path)[1].lower()
        self.raw_text = ""
        self.cleaned_text = ""
    
    def parser(self):
        return self.load().clean().get_text()
        
    def load(self):
        if self.extension == ".pdf":
            self.raw_text = self._parse_pdf()
        elif self.extension == ".txt":
            self.raw_text = self._parse_text()
        elif self.extension == ".docx":
            self.raw_text = self._parse_docx()
        else:
            raise ValueError(f"Unsupported file extension: {self.extension}")
        return self
    
    def clean(self):
        self.cleaned_text = self._clean_text(self.raw_text)
        return self
    
    def get_text(self) -> str:
        return self.cleaned_text
    
    def _parse_pdf(self) -> str:
        pdf = pymupdf.open(self.file_path)
        text = "".join([page.get_text() for page in pdf])
        return text
    
    def _parse_text(self) -> str:
        with open(self.file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _parse_docx(self) -> str:
        doc = Document(self.file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def _clean_text(self, text:str) -> str:
        # Remove leading/trailing whitespace
        text = text.strip()                     
        # Replace multiple spaces/tabs/newlines with one space               
        text = re.sub(r'\s+', ' ', text)                  
        # Replace multiple newlines with one newline
        text = re.sub(r'\n{2,}', ' \n', text)  
        return text

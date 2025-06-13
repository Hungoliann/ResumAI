import pymupdf
import os
import re

###################################################
# function will take in a file path as a string and will return a string
# Parameter: file_path (str) - file path to the resume file
# Returns: str: extracted and cleaned text for the document type
######################################################
def parse_resume(file_path:str) -> str:
    # extract the ending portion of the filepath
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        resume_text = parse_pdf(file_path)
    if ext == ".txt":
        resume_text = parse_txt(file_path)
    if ext == ".docx":
        resume_text = parse_docx(file_path)
    return resume_text

###################################################
# function will extract text from the pdf
# Parameter: file_path (str) - file path to the resume file
# Returns: extracted and cleaned text of pdf
######################################################
def parse_pdf(file_path: str) -> str:
    pdf = pymupdf.open(file_path)
    all_text = ""
    for page in pdf:
        all_text += page.get_text()
    cleaned_pdf = clean_txt(all_text)
    return cleaned_pdf

###################################################
# function will extract text from a text file
# Parameter: file_path (str) - file path to the resume file
# Returns: extracted and cleaned text of text
######################################################
def parse_txt(file_path: str) -> str:
    with open(file_path) as txt:
        while True:
            read_txt = txt.readlines()
            if not read_txt:
                break
    cleaned_txt = clean_txt(read_txt)
    return cleaned_txt

###################################################
# function will extract text from a text file
# Parameter: file_path (str) - file path to the resume file
# Returns: extracted and cleaned text of docx
######################################################
def parse_docx(file_path: str) -> str:
    doc = pymupdf.open(file_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text()
    cleaned_txt = clean_txt(all_text)
    return cleaned_txt

###################################################
# function will clean the text by removing leading/trailing whitespace, multiple spaces/tabs/newlines with one space, and replace multiple new lines with one newline
# Parameter: text(str) - text extracted from pdf, text, or docx
# Returns: cleaned text 
######################################################
def clean_txt(text:str) -> str:
    # Remove leading/trailing whitespace
    text = text.strip()                     
    # Replace multiple spaces/tabs/newlines with one space               
    text = re.sub(r'\s+', ' ', text)                  
    # Replace multiple newlines with one newline
    text = re.sub(r'\n{2,}', ' \n'. text)  
    return text

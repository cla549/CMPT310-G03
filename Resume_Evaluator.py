import pandas as pd 
import matplotlib.pyplot as plt
import re
from pypdf import PdfReader


###################################################################################################
# Simple pdf cleaner that returns it all as a string (from AI)
def pdf_to_string(pdf_path: str) -> str:
    """Extracts text from a PDF, removes symbols/formatting, and returns a single clean string of words."""
    try:
        # 1. Initialize the PDF reader
        reader = PdfReader(pdf_path)
        raw_text_list = []

        # 2. Extract text from every page
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                raw_text_list.append(page_text)

        # Combine all extracted text into one raw string
        combined_text = " ".join(raw_text_list)

        # 3. Clean the text using Regular Expressions
        # Replace hyphens/dashes with a space to avoid merging compound words
        cleaned_text = re.sub(r"[-–—]", " ", combined_text)

        # Keep only alphanumeric characters and spaces (removes bullets, punctuation, icons)
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned_text)

        # Remove single character words (e.g., 'a', 'I', 'x')
        # \b ensures we only target standalone characters, not letters inside words
        cleaned_text = re.sub(r"\b\w\b", "", cleaned_text)

        # Substitute any sequence of tabs, newlines, or multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        # 4. Remove leading/trailing spaces and return
        return cleaned_text.strip()

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return ""
###################################################################################################



# Import the Naive Bayes model trained earlier
import joblib
artifacts = joblib.load("ResuMatch.pkl")
loaded_vectorizer = artifacts["tfidf"]
ResuMatch_model = artifacts["model"]

# Import a sample resume and clean it
# Used some samples from "raw_resume" folder, renamed them for convenience
sample_title = "Sample_Resumes/Sample_Resume3.pdf"
straight_resume_text = pdf_to_string(sample_title)
#print(resume_text) # To check the output is correct

# Transform the new resume data to clear it using TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorized_text = loaded_vectorizer.transform([straight_resume_text])

# Predict the job classification of the resume
resume_prediction = ResuMatch_model.predict(vectorized_text)
print(f"The given resume would be a great {resume_prediction}.")

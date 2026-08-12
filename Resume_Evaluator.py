import numpy as np
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



# Import the model trained earlier
import joblib
artifacts = joblib.load("ResuMatch_LSV.pkl")

loaded_tfidf = artifacts["tfidf"]
ResuMatch_model = artifacts["model"]
loaded_vectorizer = artifacts["vectorizer"]

# Import full dataset
#df = pd.read_csv('data/Resume_Dataset.csv')
#dataset_sample = df["Text"][1000]
#print(dataset_sample)

# Import a sample resume and clean it
# Used some samples from "raw_resume" folder, renamed them for convenience
#sample_title = "Sample_Resumes/Sample_Resume2.pdf"
#straight_resume_text = pdf_to_string(sample_title)
#print(resume_text) # To check the output is correct

# Transform the new resume data to clear it using TFIDF
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import CountVectorizer

#vectorized_text = loaded_vectorizer.transform([straight_resume_text]) # Choose this for an imported PDF
#vectorized_text = loaded_vectorizer.transform([dataset_sample]) # Choose this for a sample from the dataset
#tfidf_text = loaded_tfidf.transform(vectorized_text)

# Labels for the jobs
job_labels = [
    "Business Analyst", 
    "Business Intelligence/Object", 
    "Datawarehousing", 
    "Java Developer", 
    "Network/Systems Admin", 
    "Project Manager", 
    "Recruiter", 
    "SQL Developer", 
    "Web Developer"
    ]

ROLE_DISPLAY_NAMES = {
    "Business Analyst (BA) Resumes": "Business Analyst",
    "Business Intelligence, Business Object Resumes": "Business Intelligence/Object",
    "Datawarehousing, ETL, Informatica Resumes": "Datawarehousing",
    "Java Developers/Architects Resumes": "Java Developer",
    "Network and Systems Administrators Resumes": "Network/Systems Admin",
    "Project Manager Resumes": "Project Manager",
    "Recruiter Resumes": "Recruiter",
    "SQL Developers Resumes": "SQL Developer",
    "Web Developer Resumes": "Web Developer",
}

# Predict the job classification of the resume
#resume_prediction = ResuMatch_model.predict(vectorized_text)[0]

# Calculate the distance of the resume from the 9 possible job options
#distance_score = ResuMatch_model.decision_function(vectorized_text)[0]


################################################################################################
# Print the results of the distance calculation and prediction
#print(f"\nThe given resume would be a great {resume_prediction}.\n")

# Using a sigmoid to calculate the percent likeness from the distance score
def sigmoid_percent(x):
    return 100 / (1 + np.exp(-x * 0.1))

#best_options = []
#worst_options = []

# Create lists of the best and worst job options
#for i in range(len(job_labels)):
#    score = float(distance_score[i])
#    if score > 0:
#        best_options.append((score, job_labels[i]))
#    else:
#        worst_options.append((score, job_labels[i]))

# Sorts the options based on the scores (high to low)
#final_best_options = sorted(best_options, reverse = True)
#final_worst_options = sorted(worst_options, reverse = True)

#print("=== Distance scores of your resume ==")
#print("\nBest options:")
#for i in range(len(final_best_options)):
#    percent = sigmoid_percent(final_best_options[i][0])
#    print(f'{i+1}. {final_best_options[i][1]} => {percent:.2f}%')
#print("\nWorst options:")
#for i in range(len(final_worst_options)):
#    percent = sigmoid_percent(final_worst_options[i][0])
#    print(f'{i+1}. {final_worst_options[i][1]} => {percent:.2f}%')

#print() # for spacing


def evaluate_role(resume_text, selected_role):
    vectorized_text = loaded_vectorizer.transform([resume_text])

    distance_scores = ResuMatch_model.decision_function(
        vectorized_text
    )[0]

    for i in range(len(ResuMatch_model.classes_)):
        model_role = ResuMatch_model.classes_[i]

        display_role = ROLE_DISPLAY_NAMES.get(
            model_role,
            model_role
        )

        if display_role == selected_role:
            score = distance_scores[i]
            return sigmoid_percent(float(score))

    raise ValueError(
        f"Unknown job role: {selected_role}"
    )

def predict_best_role(resume_text):
    vectorized_text = loaded_vectorizer.transform([resume_text])
    prediction = ResuMatch_model.predict(vectorized_text)[0]
    return ROLE_DISPLAY_NAMES.get(prediction, prediction)

def get_all_role_scores(resume_text):
    vectorized_text = loaded_vectorizer.transform([resume_text])

    distance_scores = ResuMatch_model.decision_function(
        vectorized_text
    )[0]

    results = []

    for i in range(len(ResuMatch_model.classes_)):
        model_role = ResuMatch_model.classes_[i]
        score = distance_scores[i]

        display_role = ROLE_DISPLAY_NAMES.get(
            model_role,
            model_role
        )

        percent = sigmoid_percent(float(score))

        results.append(
            (display_role, round(percent, 1))
        )

    return sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )





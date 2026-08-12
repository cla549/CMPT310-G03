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
# Model importer
import joblib

# Takes the name of the model and returns the three parameters as a list: vectorizer, TF-IDF transformation and the model parameters
def import_model(model: str):
    artifacts = joblib.load("ResuMatch_LSV.pkl")

    loaded_tfidf = artifacts["tfidf"]
    ResuMatch_model = artifacts["model"]
    loaded_vectorizer = artifacts["vectorizer"]
    return loaded_vectorizer, loaded_tfidf, ResuMatch_model
    
###################################################################################################
# Predictor
# Requires the previous functions to work

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# Actual predictor, takes the model name and the cleaned resume as a string as input, returns the prediction and a list of tuples with the distance scores of the resume for each job label
def ResuMatch_prediction(model_name: str, pdf_string: str):
    # Job Labels
    job_labels = ["Business Analyst", "Business Intelligence/Object", "Datawarehousing", "Java Developer", "Network/Systems Admin", "Project Manager", "Recruiter", "SQL Developer", "Web Developer"]

    # Converts given pdf to a clean string
    #cleaned_text = pdf_to_string(pdf_path)
    # Imports the model and all parameters needed
    loaded_vectorizer, loaded_tfidf, ResuMatch_model = import_model(model_name)
    #imported_model = import_model(model_name)
    
    # Vectorizes the data and does a TF-IDF transform as well
    vectorized_text = loaded_vectorizer.transform([pdf_string])#cleaned_text])
    tfidf_text = loaded_tfidf.transform(vectorized_text)
    
    # Predict the job classification of the resume
    resume_prediction = ResuMatch_model.predict(tfidf_text)[0]

    # Calculate the distance of the resume from the 9 possible job options
    distance_score = ResuMatch_model.decision_function(tfidf_text)[0]

    # Combine the job labels with the scores for the resume
    score_label = []
    for i in range(len(job_labels)):
        score_label.append((float(distance_score[i]), job_labels[i]))
    score_label.reverse()

    return resume_prediction, score_label


###################################################################################################
# Top Feature Extractor

# Takes the model name, cleaned resume string, the job label and the number of top words wanted.
# Returns a list of tuples with the first top_n words that are significant for the given job_label.
def resume_feat_extract(model_name: str, pdf_string: str, job_label: int, top_n):
    # Top ten initializing
    top_words_resume = []
    
    #Load model
    loaded_vectorizer, loaded_tfidf, ResuMatch_model = import_model(model_name)
    
    # Vectorize the resume
    vectorized_text = loaded_vectorizer.transform([pdf_string])#cleaned_text])
    tfidf_text = loaded_tfidf.transform(vectorized_text)
    
    # Gather feature names and flatten TF-IDF vectorized resume
    feature_names = loaded_vectorizer.get_feature_names_out()
    resume_tfidf_scores = tfidf_text.toarray().flatten()

    # Gather class coefficients of the corresponding job class and calculates word importance
    job_labels = ResuMatch_model.classes_
    job_label_idx = np.where(job_labels == job_label)[0][0]
    class_coeff = ResuMatch_model.coef_[job_label_idx]
    word_importance = class_coeff * resume_tfidf_scores

    # Gives the indices of words with scores above 0
    present_words_indices = np.where(resume_tfidf_scores > 0)[0]

    # List out all significant words with their importance and sorts them
    significant_words = []
    for idx in present_words_indices:
        significant_words.append((float(word_importance[idx]), feature_names[idx]))
    significant_words = sorted(significant_words, reverse = True)

    # Takes only the top_n words
    top_words_resume.append(significant_words[:10])
    
    return top_words_resume[0]


# Returns the top top_n best features for the given job_label with the weights
def class_feat_extract(model_name: str, job_label: int, top_n):
    top_words_class = []

    #Load model
    loaded_vectorizer, loaded_tfidf, ResuMatch_model = import_model(model_name)

    # Get the categories the model was trained on
    feature_names = loaded_vectorizer.get_feature_names_out()

    # Gather class coefficients of the corresponding job class
    job_labels = ResuMatch_model.classes_
    job_label_idx = np.where(job_labels == job_label)[0][0]
    class_coeff = ResuMatch_model.coef_[job_label_idx]

    # Sort the coefficient indices from lowest weight to highest weight
    sorted_indices = np.argsort(class_coeff)

    # Pull the top_n feature indices with the largest positive coefficients
    top_positive_indices = sorted_indices[-top_n:][::-1]
    top_words = feature_names[top_positive_indices]
    top_weights = class_coeff[top_positive_indices]

    for i in range(top_n):
        top_words_class.append((top_weights[i], top_words[i]))
    
    return top_words_class


###################################################################################################

# Using a sigmoid to calculate the percent likeness from the distance score
def sigmoid_percent(x):
    return 100 / (1 + np.exp(-x * 1))

# Takes the tuple list output of the prediction and returns a sorted tuple list of the best outcomes with the percentage score
def best_option_list(score_label):
    best_options = []
    for score, label in score_label:
        if score > 0:
            percent_score = sigmoid_percent(score)
            best_options.append((percent_score, label))
    return sorted(best_options, reverse = True)

def worst_option_list(score_label):
    worst_options = []
    for score, label in score_label:
        if score < 0:
            percent_score = sigmoid_percent(score)
            worst_options.append((percent_score, label))
    return sorted(worst_options, reverse = True)
    

###################################################################################################
# Uncomment to test the code

"""
# Test code
sample_title = "Sample_Resumes/Sample_Resume3.pdf"
cleaned_text = pdf_to_string(sample_title)
model_name = "ResuMatch_LSV.pkl"

prediction, score_label_list = ResuMatch_prediction(model_name, cleaned_text)#sample_title)
print(f"\nThe given resume would be a great {prediction}.\n")

best_list = best_option_list(score_label_list)
worst_list = worst_option_list(score_label_list)

print("=== Distance scores of your resume ==")
print("\nBest options:")
for i in range(len(best_list)):
    print(f'{i+1}. {best_list[i][1]} => {best_list[i][0]:.2f}%')
print("\nWorst options:")
for i in range(len(worst_list)):
    print(f'{i+1}. {worst_list[i][1]} => {worst_list[i][0]:.2f}%')

print(f"\nMost significant words related to {prediction}:")
best_match_words = resume_feat_extract(model_name, cleaned_text, prediction, 10)
#print(best_match_words)
index = 1
for weight, word in best_match_words:
    print(f'{index}. {word} -> {weight:.3f}')
    index += 1

optimal_job_words = class_feat_extract(model_name, "Java Developers/Architects Resumes", 20)
print()

for weight, word in optimal_job_words:
    print(f'{word} -> {weight:.3f}')


print("\nOK!\n")

"""


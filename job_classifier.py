import pandas as pd
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from data_cleaning import preprocess_resume

def train_classifier():
    raw_df = pd.read_csv("data/AI_Resume_Screening.csv")
    df = preprocess_resume(raw_df)

    x = df["Resume_Text"]
    y = df["Job Role"] 

    tfidf = TfidfVectorizer(stop_words="english")
    X_tfidf = tfidf.fit_transform(x)

    model = LinearSVC(random_state=42)
    model.fit(X_tfidf, y)

    return model, tfidf

def predict_job_role(resume_text, model, tfidf):
    resume_vector = tfidf.transform([resume_text])
    prediction = model.predict(resume_vector)[0]

    return prediction

def save_classifier(model, tfidf):
    artifacts = {
        "model": model,
        "tfidf": tfidf,
    }

    joblib.dump(
        artifacts, 
        "ResuMatch_classifier.pkl"
    )

def load_classifier():
    artifacts = joblib.load("ResuMatch_classifier.pkl")

    return artifacts["model"], artifacts["tfidf"]

if __name__ == "__main__":
    model, tfidf = train_classifier()

    save_classifier(model, tfidf)

    sample_resume = (
        "python sql machine learning pandas "
        "scikit learn 5 years b sc aws ceerified 6 projects"
    )

    prediction = predict_job_role(
        sample_resume, 
        model, 
        tfidf,
    )

    print("Predicted Job Role:", prediction)
    print("Classifier saved successfully")

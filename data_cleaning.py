"""
CMPT 310 - ResuMatch

This module preprocesses the resume and job datasets by:
- selecting relevant features
- handling missing values
- cleaning text 
- generating Resume_Text and Job_Text
- exporing cleaned datasets

"""

import numpy as np
import pandas as pd
import sys
import re 
from pathlib import Path



# ---------------------------
# Text Cleaning
# ---------------------------
def clean_text(text):
    """Convert text to lowercase and remove punctuation and extra spaces."""
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text



# ----------------------------
# Resume Preprocessing
# ----------------------------
def preprocess_resume(df):
    """Select and clean the useful resume columns."""
    resume_clean = df[
        [
            "Skills",
            "Experience (Years)",
            "Education",
            "Certifications",
            "Projects Count",
            "Job Role"
        ]
    ].copy()

    resume_clean["Certifications"] = (
        resume_clean["Certifications"].fillna("")
    )

    resume_clean["Resume_Text"] = (
        resume_clean["Skills"] 
        + " " 
        + resume_clean["Experience (Years)"].astype(str) 
        + " years "
        + resume_clean["Education"] 
        + " " 
        + resume_clean["Certifications"]
        + " "
        + resume_clean["Projects Count"].astype(str)
        + " projects"
    )

    resume_clean["Resume_Text"] = (
        resume_clean["Resume_Text"].apply(clean_text)
    )

    return resume_clean



# ----------------------------
# Job Preprocessing
# ----------------------------
def preprocess_jobs(df):
    """Select and clean the useful job-posting columns."""

    jobs_clean = df[
        [
            "Title",
            "Company",
            "Industry",
            "Job.Description",
            "Employment.Type",
            "Education.Required"
        ]
    ].copy()

    jobs_clean["Company"] = jobs_clean["Company"].fillna("Unknown")
    jobs_clean["Industry"] = jobs_clean["Industry"].fillna("Not Specified")
    jobs_clean["Job.Description"] = jobs_clean["Job.Description"].fillna("")
    jobs_clean["Employment.Type"] = (
        jobs_clean["Employment.Type"].fillna("Not Specified")
    )
    jobs_clean["Education.Required"] = (
        jobs_clean["Education.Required"].fillna("Not Specified")
    )

    jobs_clean["Job_Text"] = (
        jobs_clean["Title"] 
        + " "
        + jobs_clean["Company"]
        + " "
        + jobs_clean["Job.Description"]
    )

    jobs_clean["Job_Text"] = jobs_clean["Job_Text"].apply(clean_text)

    return jobs_clean


# ----------------------------
# Save Processed Files
# ----------------------------
def save_clean_data(df, filename):
    """Save a processed DataFrame as a CSV file."""

    df.to_csv(filename, index=False)




def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python data_cleaning.py "
            "<resume_csv> <jobs_csv>"
        )
        return
    resume_path = sys.argv[1]
    jobs_path = sys.argv[2]

    resume_df = pd.read_csv(resume_path)
    jobs_df = pd.read_csv(jobs_path)

    resume_clean = preprocess_resume(resume_df)
    jobs_clean = preprocess_jobs(jobs_df)

    Path("output").mkdir(parents=True, exist_ok=True)

    save_clean_data(
        resume_clean,
        "output/clean_resume_data.csv"
    )

    save_clean_data(
        jobs_clean,
        "output/clean_jobs_data.csv"
    )

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()

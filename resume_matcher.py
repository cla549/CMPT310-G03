from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

SKILL_LIST = [
    "python",
    "sql",
    "r",
    "java",
    "javascript",
    "c++",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "matplotlib",
    "tableau",
    "power bi",
    "excel",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "git",
    "github",
    "linux",
    "react",
    "html",
    "css",
    "nlp",
    "data analysis",
    "data visualization",
    "regression",
    "anova",
]

DISPLAY_NAMES = {
    "sql": "SQL",
    "aws": "AWS",
    "github": "GitHub",
    "git": "Git",
    "python": "Python",
    "r": "R",
    "anova": "ANOVA",
    "nlp": "NLP",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "scikit-learn",
    "c++": "C++",
    "javascript": "JavaScript",
    "html": "HTML",
    "css": "CSS",
    "power bi": "Power BI",
    "matplotlib": "Matplotlib",
    "pandas": "Pandas",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "tableau": "Tableau",
}

def format_skill_name(skill):
    return DISPLAY_NAMES.get(skill.lower(), skill.title()) 

def find_skills(text):
    text = text.lower()
    found = []

    for skill in SKILL_LIST:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text): 
            found.append(skill)

    return found

def compare_skills(resume_text, job_text):
    resume_skills = set(find_skills(resume_text))
    job_skills = set(find_skills(job_text))

    matched_skills = sorted(job_skills & resume_skills)
    missing_skills = sorted(job_skills - resume_skills)

    return matched_skills, missing_skills

def calculate_match(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words = "english")

    tfidf_matrix = vectorizer.fit_transform([
        resume_text,
        job_text
    ])

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    match_score = similarity * 100

    return match_score 



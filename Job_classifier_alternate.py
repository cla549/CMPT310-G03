# Adapted code from cla549 - job_classification.ipynb

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns #used for prope graphin
from data_cleaning import preprocess_resume #.py cleaning file for preprocess_resume function


# Import full dataset
df = pd.read_csv('data/Resume_Dataset.csv')
# Clean dataset for training
df[["category", "job_title", "Text"]].head(10)
# print(df.head(10)) # To visually inspect the first 10 elements


# Assigning categories and text
X = df['Text']
Y = df['category']
# Choose the 70/30 split to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, stratify= Y)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

""" First attempt of Vectorizing
# Removing common english words from text using TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english') #to remove english words which are common
X_train_tfidf = tfidf.fit_transform(X_train) #fittin n transform
X_test_tfidf = tfidf.transform(X_test) #transforms test set from wht we just learned earlier essentially
print(f"TF-IDF Matrix Shape: {X_train_tfidf.shape}")
print(f"Total Unique Vocabulary Learned: {len(tfidf.get_feature_names_out())} words")
"""

# NEW VECTORIZING ATTEMPT (SEE PREVIOUS)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Removes all common english words and all numerical values
count_vect = CountVectorizer(stop_words='english', token_pattern=r'(?u)\b[A-Za-z][A-Za-z]+\b')
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

tfidf = TfidfTransformer()#.fit(X_train_counts) 
X_train_tfidf = tfidf.fit_transform(X_train_counts)
X_test_tfidf = tfidf.transform(X_test_counts)
print(f"TF-IDF Matrix Shape: {X_train_tfidf.shape}")
print(f"Total Unique Vocabulary Learned: {len(tfidf.get_feature_names_out())} words")

# Applying the Multinomial Naive Bayes model to the training and fit data
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

naiveBayes = MultinomialNB() #model
naiveBayes.fit(X_train_tfidf, y_train) #test model using tfidf data
y_pred = naiveBayes.predict(X_test_tfidf) #this for prediction on the data tht we havent seen
#performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Additional testing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

logisticReg = LogisticRegression() #logisicReg model
logisticReg.fit(X_train_tfidf, y_train) 
logicRegPred = logisticReg.predict(X_test_tfidf) 
logicRegAcc = accuracy_score(y_test, logicRegPred)

svc = LinearSVC(random_state=42) #this for lin support vec machine model 
svc.fit(X_train_tfidf, y_train)
svcPred = svc.predict(X_test_tfidf)
svcAcc = accuracy_score(y_test, svcPred)

print("-- Model Comparison --")
print(f"Multinomial Naive Bayes:  {accuracy * 100:.2f}%")
print(f"Logistic Regression:      {logicRegAcc * 100:.2f}%")
print(f"Linear Support Vector:    {svcAcc * 100:.2f}%")

print("=== Classification Report (Linear Support Vector)===")
print(classification_report(y_test, svcPred))

# Visual Confusion Matrix
import seaborn as sns

#Manually rewrite labels
job_labels = ["Business Analyst", "Business Intelligence/Object", "Datawarehousing", "Java Developer", "Network/Systems Admin", "Project Manager", "Recruiter", "SQL Developer", "Web Developer"]

# Confusion Matrix for the Naive Bayes Model
confuMatrix = confusion_matrix(y_test, y_pred, labels=naiveBayes.classes_)
#plt.figure(figsize = (10, 10)) 
sns.heatmap(confuMatrix, annot=True, fmt='d', cmap='Blues', square=True, xticklabels=job_labels, yticklabels=job_labels) #heatmap plotting
plt.xticks(rotation=45, ha="right") # Angles the xticks to the right
plt.title('Confusion Matrix - Naive Bayes Classifier')
plt.xlabel('Predicted Job Categories')
plt.ylabel('Actual Job Categories')
plt.tight_layout()
plt.show()

# Confusion Matrix for the Linear Support Vector Model (LSV)
confuMatrix = confusion_matrix(y_test, svcPred, labels=naiveBayes.classes_)
#plt.figure(figsize = (10, 10)) 
sns.heatmap(confuMatrix, annot=True, fmt='d', cmap='Blues', square=True, xticklabels=job_labels, yticklabels=job_labels) #heatmap plotting
plt.xticks(rotation=45, ha="right") # Angles the xticks to the right
plt.title('Confusion Matrix - Linear Support Vector Model]')
plt.xlabel('Predicted Job Categories')
plt.ylabel('Actual Job Categories')
plt.tight_layout()
plt.show()

##################################################################################################
# Uncomment to save the model and vectorizer to disk

import joblib

artifacts = {
    "vectorizer": count_vect,
    "tfidf": tfidf,
    "model": svc
}
joblib.dump(artifacts, "ResuMatch_LSV.pkl")


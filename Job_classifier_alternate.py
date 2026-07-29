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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, stratify= Y)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Removing common english words from text using TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english') #to remove english words which are common
X_train_tfidf = tfidf.fit_transform(X_train) #fittin n transform
X_test_tfidf = tfidf.transform(X_test) #transforms test set from wht we just learned earlier essentially
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

# Visual Confusion Matrix
import seaborn as sns

confuMatrix = confusion_matrix(y_test, y_pred, labels=naiveBayes.classes_)

plt.figure(figsize=(14, 14)) #heatmap plotting
sns.heatmap(confuMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=naiveBayes.classes_, yticklabels=naiveBayes.classes_)
plt.title('Confusion Matrix - Naive Bayes Classifier')
plt.xlabel('Predicted Job Role')
plt.ylabel('Actual Job Role')
#plt.tight_layout()
plt.show()


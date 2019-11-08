import numpy as numpy
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
import spacy
from spacy.lemmatizer import Lemmatizer
nlp = spacy.load("en_core_web_sm")
import pandas as pd
import shutil

regression_data = pd.read_csv('./total_fasc.csv', encoding = "ISO-8859-1")
regression_data = regression_data.dropna(subset=['Com_first_resp'])
regression_data = regression_data.dropna(subset=['response_text'])
regression_data = regression_data.loc[regression_data['condition_value'] == 'B']

df = pd.DataFrame(regression_data, columns = ['Com_first_resp', 'response_text_spell-checked'])
pos = df['Com_first_resp'] = 1
neg = df['Com_first_resp'] = 0

for row in regression_data:
  print(row)
for i, row in enumerate(regression_data.Com_first_resp):
  if row == 1:
    for x, text in enumerate(regression_data.response_text):
      if (i == x):
        file = open('./response_items/pos/response_' + str(i) + '.txt', 'w+')
        file.write(text)
  if row == 0:
    for x, text in enumerate(regression_data.response_text):
      if (i == x):
        file = open('./response_items/neg/response_' + str(i) + '.txt', 'w+')
        file.write(text)

raw_data = load_files('./response_items')
X, y = raw_data.data, raw_data.target
print(y)
# true_false = [1, 1, 1]
# X = raw_text
# y = true_false

documents = []

from nltk.stem import WordNetLemmatizer

# stemmer = WordNetLemmatizer()
# feelings = stemmer.lemmatize('feelings')
# print(feelings)
for sen in range(0, len(X)):
    # Remove all the special characters
    # document = re.sub(r'\W', ' ', str(X[sen]))

    
    # Converting to Lowercase
    document = str(X[sen]).lower()
    
    # Lemmatization
    document = nlp(document)
    document = [word.lemma_ for word in document]
    # document = document.split()

    # document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    document = re.sub(r'\W', ' ', document)
        # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    ## by removing pronoun we increase accuracy by 3-4%
    document = re.sub('PRON', '', document)
    
    documents.append(document)

print(documents)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=1, max_df=0.7, ngram_range=(1,3), stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# randomforest is the first one i tried
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(n_estimators=1000, random_state=0, learning_rate=1.0, max_depth=3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
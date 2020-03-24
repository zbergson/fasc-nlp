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
import os
import glob

regression_data = pd.read_csv('./fasc_data/FASC_S17_F17_S18_all_Rounds_12-7-19_CHECKED.csv', encoding = "ISO-8859-1")
regression_data = regression_data.dropna(subset=['Com_first_resp'])
regression_data = regression_data.dropna(subset=['spelling_checked_response_text'])
regression_data = regression_data.loc[regression_data['condition_value'] == 'H']
print(len(regression_data))
# regression_data = regression_data.head(200)
current_condition = 'H'

df = pd.DataFrame(regression_data, columns = ['Com_first_resp', 'spelling_checked_response_text'])
pos = df['Com_first_resp'] = 1
neg = df['Com_first_resp'] = 0

training_set = regression_data
# validation_set = regression_data[-100:]

# randomntArray = ['2908', '3115', '3111', '3112', '3110', '3103', '3104']
# regression_data = regression_data[~regression_data['username'].isin(randomntArray)]
# asdArray = ['2203',
# '2204',
# '2206',
# '2207',
# '2209',
# '2210'
# '2211',
# '2213',
# '2214',
# '2215',
# '2216',
# '2217',
# '2301',
# '2303',
# '2304',
# '2306',
# '2307',
# '2308',
# '2309',
# '2310',
# '2311',
# '2312',
# '2313',
# '2314',
# '2315',
# '2316']
# regression_data = regression_data[~regression_data['username'].isin(asdArray)]
# value = regression_data.loc[regression_data['username'] == '2908']
# print(value, 'tony')
# regression_data_check = regression_data.username.isin(['2316', '2315'])
# print(regression_data_check)
# for row in regression_data:
#   print(row)
# for i, row in enumerate(training_set.Com_first_resp):
#   if row == 1:
#     for x, text in enumerate(training_set.spelling_checked_response_text):
#       if (i == x):
#         file = open('./response_items/pos/response_' + str(i) + '.txt', 'w+')
#         file.write(text)
#   if row == 0:
#     for x, text in enumerate(training_set.spelling_checked_response_text):
#       if (i == x):
#         file = open('./response_items/neg/response_' + str(i) + '.txt', 'w+')
#         file.write(text)

# raw_data = load_files('./response_items')
# print(raw_data)

X = training_set.spelling_checked_response_text.tolist()
y = training_set.Com_first_resp.tolist()

# X, y = raw_data.data, raw_data.target
# y = y[0:320]
# z_response = y[-100:]
# Z_test = X[-100:]
# X = X[0:320]
# print(y)
# true_false = [1, 1, 1]
# X = raw_text
# y = true_false

documents = []
documents_z = []

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
    document = document.rstrip()
    document = " ".join(document.split())
    documents.append(document)

# for sen in range(0, len(Z_test)):
#     # Remove all the special characters
#     # document = re.sub(r'\W', ' ', str(X[sen]))

    
#     # Converting to Lowercase
#     document_z = str(Z_test[sen]).lower()
    
#     # Lemmatization
#     document_z = nlp(document_z)
#     document_z = [word.lemma_ for word in document_z]
#     # document = document.split()

#     # document = [stemmer.lemmatize(word) for word in document]
#     document_z = ' '.join(document_z)
#     document_z = re.sub(r'\W', ' ', document_z)
#         # remove all single characters
#     document_z = re.sub(r'\s+[a-zA-Z]\s+', ' ', document_z)
    
#     # Remove single characters from the start
#     document_z = re.sub(r'\^[a-zA-Z]\s+', ' ', document_z) 
    
#     # Substituting multiple spaces with single space
#     document_z = re.sub(r'\s+', ' ', document_z, flags=re.I)
    
#     # Removing prefixed 'b'
#     document_z = re.sub(r'^b\s+', '', document_z)

#     ## by removing pronoun we increase accuracy by 3-4%
#     document_z = re.sub('PRON', '', document_z)
#     document_z = document_z.rstrip()
#     document_z = " ".join(document_z.split())
#     documents_z.append(document_z)

print(documents)
# print(documents_z)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000, min_df=1, max_df=0.7, ngram_range=(1,3), stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()
# Z_test = vectorizer.fit_transform(documents_z).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
# Z_test = tfidfconverter.fit_transform(Z_test).toarray()

# y_train = y[0:336]
# y_test = y[-84:]
# X_train = X[0:336]
# X_test = X[-84:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, training_set_train, training_set_test = train_test_split(X, y, training_set, test_size = 0.2, random_state = 0)

# # randomforest is the first one i tried
# from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.naive_bayes import ComplementNB
# from sklearn.neural_network import MLPClassifier
# # classifier = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
classifier = GradientBoostingClassifier(n_estimators=1000, random_state=0, learning_rate=1.0, max_depth=3)
# classifier = GradientBoostingClassifier(n_estimators=1000, random_state=0, learning_rate=1.0, max_depth=3, validation_fraction=0.5, n_iter_no_change=20, tol=0.01)
# classifier = LinearSVC(C=0.01)
# classifier = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)
classifier.fit(X_train, y_train)

classifier.score(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

print(training_set_test)
print(len(training_set_test))
print(training_set_test.username)

list_of_predictions = list(y_pred)
length_of_predictions = len(list_of_predictions)
print(list_of_predictions)
print(y_test)

print(len(list_of_predictions))
regression_data_test_new = training_set_test
list_human = regression_data_test_new.Com_first_resp.tolist()
print(list_human)

regression_data_test_new['Com_first_resp'] = list_of_predictions

if (current_condition == 'A'):
  regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_a.csv', encoding='utf-8', index=False)
if (current_condition == 'B'):
  regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_b.csv', encoding='utf-8', index=False)
if (current_condition == 'C'):
  regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_c.csv', encoding='utf-8', index=False)
if (current_condition == 'D'):
  regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_d.csv', encoding='utf-8', index=False)
if (current_condition == 'E'):
  regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_e.csv', encoding='utf-8', index=False)
if (current_condition == 'F'):
  regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_f.csv', encoding='utf-8', index=False)
if (current_condition == 'G'):
  regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_g.csv', encoding='utf-8', index=False)
if (current_condition == 'H'):
  regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_h.csv', encoding='utf-8', index=False)



mismatch = []
for i in range(0, length_of_predictions):
  if (list_of_predictions[i] != list_human[i]):
    mismatch.append(1)

print(mismatch)


os.chdir("./fasc_data/machine_scores")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], sort=False)
combined_csv.to_csv( "combined_machine_scores.csv", index=False, encoding='utf-8-sig')

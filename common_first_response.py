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
import np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

current_condition = 'G'
test_reliability = False

regression_data = pd.read_csv('./fasc_data/FASC_S17_F17_S18_all_Rounds_12-7-19_CHECKED.csv', encoding = "ISO-8859-1")
regression_data = regression_data[~regression_data['username'].isin([2201, 2212, 2302, 2305, 0, 3218, 2305, 3105, 3018])]
regression_data = regression_data.dropna(subset=['Com_first_resp'])
regression_data = regression_data.dropna(subset=['spelling_checked_response_text'])
regression_data = regression_data.loc[regression_data['condition_value'] == current_condition]
print(len(regression_data))
# test_set_s17 = regression_data.loc[regression_data['Year'] == 'S17'].sample(frac=0.2, random_state=0)
# regression_data = regression_data.drop(test_set_s17.index)
print(len(regression_data))
# test_set_s18 = regression_data.loc[regression_data['Year'] == 'F17_S18'].sample(frac=0.2, random_state=0)
# regression_data = regression_data.drop(test_set_s18.index)
print(len(regression_data))

print(len(regression_data))
regression_data_length = len(regression_data)
# regression_data = regression_data.head(200)


# #test s17 only
# test_set = test_set_s17
# test_set = test_set.loc[test_set['condition_value'] == current_condition]

#test s18 only
# test_set = test_set_s18
# test_set = test_set.loc[test_set['condition_value'] == current_condition]
# print(len(test_set))

##fresh data from s19, combined with s18
# test_set = pd.read_csv('./fasc_data/FASC_S19_cleaned_SC_reliability_CHECKED.csv', encoding = "ISO-8859-1")
# test_set = test_set.loc[test_set['Round'] == 1]
# test_set = test_set.loc[test_set['condition_value'] == current_condition]
# test_set = test_set.dropna(subset=['Com_first_resp'])
# test_set_s18 = test_set_s18.loc[test_set_s18['condition_value'] == current_condition]
# frames = [test_set, test_set_s18]
# test_set = pd.concat(frames)
# print(len(test_set))

#s19 alone
test_set = pd.read_csv('./fasc_data/FASC_S19_cleaned_SC_reliability_CHECKED.csv', encoding = "ISO-8859-1")
test_username = pd.read_csv('./fasc_data/Combined_S19_Data_7-21-20.csv', encoding = "ISO-8859-1")
test_username = list(test_username['username'])
test_set = test_set[test_set['username'].isin(test_username)]

test_set = test_set.loc[test_set['condition_value'] == current_condition]
test_set = test_set.dropna(subset=['spelling_checked_response_text'])
test_set = test_set.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# test_set = test_set.dropna(subset=['Com_first_resp'])
test_set_reliability = test_set.loc[test_set['Round'] == 1]
test_set_reliability = test_set_reliability.dropna(subset=['Com_first_resp'])
print(len(test_set), 'test set length')

if test_reliability:
    test_set = test_set_reliability
else:
    test_set = test_set
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
#s17
Z = test_set.spelling_checked_response_text.tolist()
v = test_set.Com_first_resp.tolist()
#s19
# Z = test_set.response_text_SC.tolist()
# v = test_set.Com_first_resp.tolist()
print(v)
# v = v.to_numeric(v)

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

for sen in range(0, len(Z)):
    # Remove all the special characters
    # document = re.sub(r'\W', ' ', str(X[sen]))

    
    # Converting to Lowercase
    document = str(Z[sen]).lower()
    
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
    documents_z.append(document)

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

# print(documents)
print(documents_z)
from sklearn.pipeline import Pipeline

#A
#vect_max_df = 0.5
#vect_max_features = 500
#vect_ngram_range = 1,2
#clf_n_estimators = default
#clf_learning_rate = default
#clf_max_depth = default
#use_idf = True
if current_condition == 'A':
  text_clf = Pipeline([
    ('vect', CountVectorizer(max_df=0.5, max_features=400, ngram_range=(1,2), stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', GradientBoostingClassifier(learning_rate=0.25, max_depth=3, n_estimators=600, random_state=0))
  ])

#B
#vect_max_df = 0.4
#vect_max_features=1100
#vect_ngram_range = 1,3
#clf_learning_rate = default
#clf_n_estimators = default
#use_idf = True
#clf_max_depth = 5

if current_condition == 'B':
  text_clf = Pipeline([
    ('vect', CountVectorizer(max_df=0.4, max_features=1100, ngram_range=(1,3), stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', GradientBoostingClassifier(max_depth=5, random_state=0))
  ])

#C
#vect_max_df=0.6
#vect_max_features=100
#vect_ngram_range=(1,2)
#clf_learning_rate=0.05
#clf_n_estimators=300
#use_idf=True
#clf_max_depth=default
#max_features_clf = 5

if current_condition == 'C':
  text_clf = Pipeline([
    ('vect', CountVectorizer(max_df=0.6, max_features=100, ngram_range=(1,2), stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', GradientBoostingClassifier(learning_rate=0.05, max_features=5, n_estimators=300, random_state=0))
  ])

#D
#clf_max_depth=4.0

if current_condition == 'D':
  text_clf = Pipeline([
    ('vect', CountVectorizer(stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', GradientBoostingClassifier(max_depth=4, random_state=0))
  ])

#E
#vect_max_df=0.9
#vect_max_features=1800
#vect_ngram_range=(1,3)
#clf_learning_rate=0.25
#clf_n_estimators=800
#use_idf=False
#clf_max_depth=default

if current_condition == 'E':
  text_clf = Pipeline([
    ('vect', CountVectorizer(max_df=0.9, max_features=1800, ngram_range=(1,3), stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer(use_idf=False)),
    ('clf', GradientBoostingClassifier(learning_rate=0.25, n_estimators=800, random_state=0))
  ])

#F
#vect_max_df=0.5
#vect_max_features=1100
#vect_ngram_range=1,2
#clf_learning_rate=default
#clf_n_estimators=default
#use_idf=True
#clf_max_depth=2

if current_condition == 'F':
  text_clf = Pipeline([
    ('vect', CountVectorizer(max_df=0.5, max_features=1100, ngram_range=(1,2), stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', GradientBoostingClassifier(max_depth=2, random_state=0))
  ])

#G
#vect_max_df=0.6
#vect_max_features=200
#vect_ngram_range=1,2
#clf__learning_rate: 0.05
##clf__n_estimators: 300
#tfidf__use_idf: True

if current_condition == 'G':
  text_clf = Pipeline([
    ('vect', CountVectorizer(max_df=0.6, max_features=200, ngram_range=(1,2), stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, random_state=0))
  ])

#H
#vect_max_df = 0.5
#vect__max_features = 400
#vect_ngram_range = 1,2
#clf_learning_rate = 0.25
#clf_n_estimators = 600
#clf_max_depth = 3 (default)
#tfidf__use_idf = True

if current_condition == 'H':
  text_clf = Pipeline([
    ('vect', CountVectorizer(max_df=0.5, max_features=400, ngram_range=(1,2), stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', GradientBoostingClassifier(learning_rate=0.25, n_estimators=600, max_depth=3, random_state=0))
  ])
parameters = {
#   'vect__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
#   'vect__max_features': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
#   'vect__max_df': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#   'tfidf__use_idf': (True, False),
#   'clf__n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
#   'clf__learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01]
    # 'clf__max_depth': np.linspace(1, 32, 32, endpoint=True),
    # 'clf__max_features': list(range(1,regression_data.shape[1]))
}
# scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1, scoring=scoring, refit='AUC')
# gs_clf.fit(X, y)
# predicted = gs_clf.predict(Z)
# print(np.mean(predicted == v))
# for param_name in sorted(parameters.keys()):
#   print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


# text
text_clf.fit(X, y)
v_pred = text_clf.predict(Z)
print(v_pred)
if test_reliability:
  print(np.mean(v_pred == v))
  from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
  print(v_pred)
  print(confusion_matrix(v,v_pred))
  print(classification_report(v,v_pred))
  print(accuracy_score(v, v_pred))

# max_features = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
# train_results = []
# test_results = []
# for max_feature in max_features:
#   vectorizer = CountVectorizer(max_features=max_feature, ngram_range=(1,3), stop_words=stopwords.words('english'))
#   X = vectorizer.fit_transform(documents).toarray()
#   X_model = vectorizer.fit(documents)
#   # print(X.vocabulary_)
#   vectorizer_new = CountVectorizer(decode_error='replace',max_features=max_feature, ngram_range=(1,3), vocabulary=X_model.vocabulary_)
#   Z = vectorizer_new.fit_transform(documents_z).toarray()
#   from sklearn.feature_extraction.text import TfidfTransformer
#   tfidfconverter = TfidfTransformer()
#   X = tfidfconverter.fit_transform(X).toarray()
#   Z = tfidfconverter.fit_transform(Z).toarray()
#   model = GradientBoostingClassifier()
#   model.fit(X, y)
#   train_pred = model.predict(X)
#   false_positive_rate, true_positive_rate, thresholds = roc_curve(y, train_pred)
#   roc_auc = auc(false_positive_rate, true_positive_rate)
#   train_results.append(roc_auc)
#   v_pred = model.predict(Z)
#   false_positive_rate, true_positive_rate, thresholds = roc_curve(v, v_pred)
#   roc_auc = auc(false_positive_rate, true_positive_rate)
#   test_results.append(roc_auc)
# from matplotlib.legend_handler import HandlerLine2D
# line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
# line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('AUC score')
# plt.xlabel('max features')
# plt.show()



from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(max_features=500, max_df=0.5, ngram_range=(1,2), stop_words=stopwords.words('english'))
# X = vectorizer.fit_transform(documents).toarray()
# X_model = vectorizer.fit(documents)
# # print(X.vocabulary_)
# vectorizer_new = CountVectorizer(decode_error='replace',max_features=500, max_df=0.5, ngram_range=(1,2), stop_words=stopwords.words('english'), vocabulary=X_model.vocabulary_)
# Z = vectorizer_new.fit_transform(documents_z).toarray()
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidfconverter = TfidfTransformer(use_idf=True)
# X = tfidfconverter.fit_transform(X).toarray()
# Z = tfidfconverter.fit_transform(Z).toarray()


from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test, training_set_train, training_set_test = train_test_split(X, y, training_set, test_size = 0.2, random_state = 0)
# Z_train, Z_test, v_train, v_test, training_set_train, test_set_test = train_test_split(Z, v, test_set, test_size = 0.2, random_state = 0)

# # randomforest is the first one i tried
# from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.naive_bayes import ComplementNB
# from sklearn.neural_network import MLPClassifier
# # classifier = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
classifier = GradientBoostingClassifier()
# classifier = GradientBoostingClassifier(n_estimators=1000, random_state=0, learning_rate=1.0, max_depth=3, validation_fraction=0.5, n_iter_no_change=20, tol=0.01)
# classifier = LinearSVC(C=0.01)
# classifier = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

# model = GradientBoostingClassifier()
# model.fit(X, y)
# v_pred = model.predict(Z)
# from sklearn.metrics import roc_curve, auc
# false_positive_rate, true_positive_rate, thresholds = roc_curve(v, v_pred)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# print(roc_auc)




# classifier.fit(X, y)

# classifier.score(X, y)
# classifier.score(Z_train, v_train)

# y_pred = classifier.predict(X_test)
# v_pred = classifier.predict(Z)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))

# print(training_set_test)
# print(len(training_set_test))
# print(training_set_test.username)

# list_of_predictions = list(y_pred)
# length_of_predictions = len(list_of_predictions)
# print(list_of_predictions)
# print(y_test)

# print(len(list_of_predictions))
# regression_data_test_new = training_set_test
# list_human = regression_data_test_new.Com_first_resp.tolist()
# print(list_human)

#for fresh data
# print(v_pred)
# print(confusion_matrix(v,v_pred))
# print(classification_report(v,v_pred))
# print(accuracy_score(v, v_pred))

# # print(test_set_test)
# # print(len(test_set_test))
# # print(test_set_test.username)

# list_of_predictions_v = list(v_pred)
# length_of_predictions_v = len(list_of_predictions_v)
# print(list_of_predictions_v)
# # print(v)

# # print(len(list_of_predictions_v))
# # regression_data_test_new_v = v
# # list_human_v = regression_data_test_new_v.Com_first_resp.tolist()
# print(v)

# regression_data_test_new['Com_first_resp'] = list_of_predictions

regression_data_test_new = test_set
regression_data_test_new['Com_first_resp_machine'] = list(v_pred)
if test_reliability != True:
  if (current_condition == 'A'):
    regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_a.csv', encoding='utf-8', index=False)
  #   training_set_train.to_csv('./fasc_data/machine_scores/condition_a_train.csv', encoding='utf-8', index=False)
  if (current_condition == 'B'):
    regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_b.csv', encoding='utf-8', index=False)
  #   training_set_train.to_csv('./fasc_data/machine_scores/condition_b_train.csv', encoding='utf-8', index=False)
  if (current_condition == 'C'):
    regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_c.csv', encoding='utf-8', index=False)
  #   training_set_train.to_csv('./fasc_data/machine_scores/condition_c_train.csv', encoding='utf-8', index=False)
  if (current_condition == 'D'):
    regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_d.csv', encoding='utf-8', index=False)
  #   training_set_train.to_csv('./fasc_data/machine_scores/condition_d_train.csv', encoding='utf-8', index=False)
  if (current_condition == 'E'):
    regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_e.csv', encoding='utf-8', index=False)
  #   training_set_train.to_csv('./fasc_data/machine_scores/condition_e_train.csv', encoding='utf-8', index=False)
  if (current_condition == 'F'):
    regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_f.csv', encoding='utf-8', index=False)
  #   training_set_train.to_csv('./fasc_data/machine_scores/condition_f_train.csv', encoding='utf-8', index=False)
  if (current_condition == 'G'):
    regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_g.csv', encoding='utf-8', index=False)
  #   training_set_train.to_csv('./fasc_data/machine_scores/condition_g_train.csv', encoding='utf-8', index=False)
  if (current_condition == 'H'):
    regression_data_test_new.to_csv('./fasc_data/machine_scores/condition_h.csv', encoding='utf-8', index=False)
  #   training_set_train.to_csv('./fasc_data/machine_scores/condition_h_train.csv', encoding='utf-8', index=False)



# mismatch = []
# for i in range(0, length_of_predictions):
#   if (list_of_predictions[i] != list_human[i]):
#     mismatch.append(1)

# print(mismatch)


os.chdir("./fasc_data/machine_scores")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], sort=False)
combined_csv.to_csv( "combined_machine_scores.csv", index=False, encoding='utf-8-sig')

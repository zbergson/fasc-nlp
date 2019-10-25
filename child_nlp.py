import spacy
from spacy.matcher import PhraseMatcher
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
nlp = spacy.load("en_core_web_sm")
import csv
import pandas as pd
import numpy as np

phrase_matcher = PhraseMatcher(nlp.vocab)

noun_terms = []
with open('./Noun_Lemmatized.txt', 'r') as wordDoc:
    for line in wordDoc:
      currentPlace = line[:-1]
      currentPlace = currentPlace.replace(u'\xa0', u'')
      noun_terms.append(currentPlace.lower())

verb_terms = []
with open('./Verb_Lemmatized.txt', 'r') as wordDoc:
    for line in wordDoc:
      currentPlace = line[:-1]
      currentPlace = currentPlace.replace(u'\xa0', u'')
      verb_terms.append(currentPlace.lower())

adjective_terms = []
with open('./Adj_Lemmatized.txt', 'r') as wordDoc:
    for line in wordDoc:
      currentPlace = line[:-1]
      currentPlace = currentPlace.replace(u'\xa0', u'')
      adjective_terms.append(currentPlace.lower())

adverb_terms = []
with open('./AdVerb_Lemmatized.txt', 'r') as wordDoc:
    for line in wordDoc:
      currentPlace = line[:-1]
      currentPlace = currentPlace.replace(u'\xa0', u'')
      adverb_terms.append(currentPlace.lower())

phrase_terms = []
with open('./Phrase_Lemmatized.txt', 'r') as wordDoc:
    for line in wordDoc:
      currentPlace = line[:-1]
      currentPlace = currentPlace.replace(u'\xa0', u'')
      phrase_terms.append(currentPlace)


regression_data = pd.read_csv('./regression_2_fasc.csv')
regression_data = regression_data.dropna(subset=['Mental_terms_tot'])
total_mental = regression_data['Mental_terms_tot'].sum()
phrase_patterns = [nlp.make_doc(phrase_text) for phrase_text in phrase_terms]

phrase_matcher.add("TerminologyListPhrase", None, *phrase_patterns)
regression_total_count = 0
response_text = regression_data.response_text.replace(np.nan, '', regex=True)
regression_count_new_col = []
for sentence in response_text:
  doc = nlp(sentence)
  nouns = []
  verbs = []
  adjectives = []
  adverbs = []
  for token in doc: 
    if token.pos_ == 'NOUN':
      nouns.append(token.lemma_)
    if token.pos_ == 'ADV':
      adverbs.append(token.lemma_)
    if token.pos_ == 'VERB':
      verbs.append(token.lemma_)
    if token.pos_ == 'ADJ':
      adjectives.append(token.lemma_)
  ##figure out a way to count mental term twice if it shows up twice
  adj_intersect = set(adjective_terms).intersection(adjectives)
  adjectives_count = 0
  for word in adj_intersect:
    count = adjectives.count(word)
    adjectives_count = adjectives_count + count
  # print(adjectives_count, 'adj count')
  # print(adj_intersect, 'adjective')

  verb_intersect = set(verb_terms).intersection(verbs)
  verbs_count = 0
  for word in verb_intersect:
    count = verbs.count(word)
    verbs_count = verbs_count + count
  # print(verbs_count, 'verb count')
  # print(verb_intersect, 'verb')

  noun_intersect = set(noun_terms).intersection(nouns)
  noun_count = 0
  for word in noun_intersect:
    count = nouns.count(word)
    noun_count = noun_count + count
  # print(noun_count, 'noun count')
  # print(noun_intersect, 'noun')

  adverb_intersect = set(adverb_terms).intersection(adverbs)
  adverb_count = 0
  for word in adverb_intersect:
    count = adverbs.count(word)
    adverb_count = adverb_count + count
  # print(adverb_count, 'adverb count')
  # print(adverb_intersect, 'adverb')

  phrase_matches = phrase_matcher(doc)
  phrase_array = []
  for match_id, start, end in phrase_matches:
      span = doc[start:end]
      phrase_array.append(span.text)
  phrase_count = len(phrase_array)
  # print(phrase_count)
  
  total_count = phrase_count + adverb_count + noun_count + verbs_count + adjectives_count
  # print(total_count)
  regression_count_new_col.append(total_count)
  regression_total_count = regression_total_count + total_count

regression_data['nlp_count'] = regression_count_new_col
print(regression_total_count)
print(total_mental)
regression_data.to_csv('regression_2_fasc_nlp_analysis.csv', encoding='utf-8', index=False)
nlp_count_arr = []
mental_terms_tot_arr = []
for row in regression_data.nlp_count:
  nlp_count_arr.append(int(row))

for row in regression_data.Mental_terms_tot:
  mental_terms_tot_arr.append(int(row))


num_wrong = 0
for i, num in enumerate(nlp_count_arr):
  num_for_mental = mental_terms_tot_arr[i]
  if (num_for_mental != num):
    num_wrong = num_wrong + 1
print(num_wrong)
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


# x = pd.read_csv('~/documents/F17_S18_FASCPILOT2_Coded.csv', encoding = "ISO-8859-1")

# y = pd.read_csv('./fasc_data/regression_2_fasc.csv')

# common = x.merge(y, on=['responseID'])
# x = x[(~x.responseID.isin(common.responseID))]

# print(x)

# x.to_csv('./fasc_data/remaining_80_percent.csv')


regression_data = pd.read_csv('./fasc_data/20%_regression_2_fasc_nlp_analysis.csv')
regression_data = regression_data.dropna(subset=['response_text'])
regression_data = regression_data[~regression_data['username'].isin(['2308', '2302','2914','2913','2916',
'2918','2917','2908','3015','3026'])]
regression_data = regression_data.dropna(subset=['mental_terms_count_F19'])
#regression_data['mental_terms_count_F19'] = regression_data.Mental_terms_tot.fillna('')

## Add this back in once NA's are gone ####
total_mental = regression_data['mental_terms_count_F19'].sum()
########################################
phrase_patterns = [nlp.make_doc(phrase_text) for phrase_text in phrase_terms]

phrase_matcher.add("TerminologyListPhrase", None, *phrase_patterns)
regression_total_count = 0
response_text = regression_data.response_text.replace(np.nan, '', regex=True)
regression_count_new_col = []
regression_adjective_list = []
regression_verb_list = []
regression_noun_list = []
regression_adverb_list = []
regression_phrase_list = []
for sentence in response_text:
  doc = nlp(sentence)
  nouns = []
  verbs = []
  adjectives = []
  adverbs = []
  for i, token in enumerate(doc): 
    if token.pos_ == 'NOUN':
      nouns.append(token.lemma_)
    if token.pos_ == 'ADV':
      adverbs.append(token.lemma_)
    if token.pos_ == 'VERB':
      if token.lemma_ == 'think':
        # if word preceding think is I, don't count
        if (str(doc[i - 1]) != 'i' and str(doc[i - 1]) != 'I'):
          verbs.append(token.lemma_)
      else:
        verbs.append(token.lemma_)
    if token.pos_ == 'ADJ':
      # if (str(doc[i]) == 'bad'):
      #   adjectives.append(token.lemma_)
      if (str(doc[i - 1]) != 'feel'):
        adjectives.append(token.lemma_)
  ##figure out a way to count mental term twice if it shows up twice
  adj_intersect = set(adjective_terms).intersection(adjectives)
  adjectives_count = 0
  if len(adj_intersect) > 0:
    regression_adjective_list.append(adj_intersect)
  else:
    regression_adjective_list.append('')
  for word in adj_intersect:
    count = adjectives.count(word)
    adjectives_count = adjectives_count + count
  # print(adjectives_count, 'adj count')
  # print(adj_intersect, 'adjective')

  verb_intersect = set(verb_terms).intersection(verbs)
  # print(verbs)
  if len(verb_intersect) > 0:
    regression_verb_list.append(verb_intersect)
  else:
    regression_verb_list.append('')
  verbs_count = 0
  for word in verb_intersect:
    count = verbs.count(word)
    verbs_count = verbs_count + count
  # print(verbs_count, 'verb count')
  # print(verb_intersect, 'verb')

  noun_intersect = set(noun_terms).intersection(nouns)
  if len(noun_intersect) > 0:
    regression_noun_list.append(noun_intersect)
  else:
    regression_noun_list.append('')
  noun_count = 0
  for word in noun_intersect:
    count = nouns.count(word)
    noun_count = noun_count + count
  # print(noun_count, 'noun count')
  # print(noun_intersect, 'noun')

  adverb_intersect = set(adverb_terms).intersection(adverbs)
  if len(adverb_intersect) > 0:
    regression_adverb_list.append(adverb_intersect)
  else:
    regression_adverb_list.append('')
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
  regression_phrase_list.append(' + '.join(phrase_array))
  # print(phrase_count)
  
  total_count = phrase_count + adverb_count + noun_count + verbs_count + adjectives_count
  # print(total_count)
  regression_count_new_col.append(total_count)
  regression_total_count = regression_total_count + total_count

regression_data['nlp_count'] = regression_count_new_col
regression_data['adjective_list'] = regression_adjective_list
regression_data['verb_list'] = regression_verb_list
regression_data['noun_list'] = regression_noun_list
regression_data['adverb_list'] = regression_adverb_list
regression_data['phrase_list'] = regression_phrase_list
regression_data.to_csv('./fasc_data/regression_2_fasc_nlp_analysis.csv', encoding='utf-8', index=False)
nlp_count_arr = []
mental_terms_tot_arr = []
for row in regression_data.nlp_count:
  nlp_count_arr.append(int(row))

can_do_comparison = True
for row in regression_data.mental_terms_count_F19:
  if (pd.isna(row)):
    can_do_comparison = False
    #print("can't calculate b/c NA exists in column. Please count mental terms for comparison")
  else:
    mental_terms_tot_arr.append(int(row))


num_wrong = 0
if (can_do_comparison):
  for i, num in enumerate(nlp_count_arr):
    num_for_mental = mental_terms_tot_arr[i]
    if (num_for_mental != num):
      num_wrong = num_wrong + 1
  print(num_wrong)
else:
  print('must get rid of NA in mental terms total column to do this')

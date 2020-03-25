import spacy
from spacy.matcher import PhraseMatcher
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
nlp = spacy.load("en_core_web_sm")
import csv
import pandas as pd
import numpy as np
import os
import glob

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


regression_data = pd.read_csv('./fasc_data/FASC_S17_F17_S18_all_Rounds_12-7-19_CHECKED.csv', encoding = "ISO-8859-1")
regression_data_other_rounds_and_asd = pd.read_csv('./fasc_data/FASC_S17_F17_S18_all_Rounds_12-7-19_CHECKED.csv', encoding = "ISO-8859-1")
regression_data_round_three_ASD_1 = pd.read_csv('./fasc_data/FASC_S17_F17_S18_all_Rounds_12-7-19_CHECKED.csv', encoding = "ISO-8859-1")

regression_data = regression_data.dropna(subset=['spelling_checked_response_text'])
regression_data = regression_data.loc[regression_data['ASD'] == 0]
regression_data = regression_data.loc[regression_data['Round'] == 3]
regression_data = regression_data.dropna(subset=['Mental_terms_tot_2'])

regression_data_other_rounds_and_asd = regression_data_other_rounds_and_asd.loc[regression_data_other_rounds_and_asd['ASD'] == 0]
regression_data_other_rounds_and_asd = regression_data_other_rounds_and_asd.loc[(regression_data_other_rounds_and_asd['Round'] > 3) | (regression_data_other_rounds_and_asd['Round'] < 3)]

regression_data_round_three_ASD_1 = regression_data_round_three_ASD_1.loc[regression_data_round_three_ASD_1['ASD'] == 1]
regression_data_round_three_ASD_1 = regression_data_round_three_ASD_1.loc[regression_data_round_three_ASD_1['Round'] >= 0]
print(len(regression_data), 'regression_data')
print(len(regression_data_other_rounds_and_asd), 'second')
print(len(regression_data_round_three_ASD_1), 'third')
#regression_data['mental_terms_count_F19'] = regression_data.Mental_terms_tot.fillna('')

## Add this back in once NA's are gone ####
total_mental = regression_data['Mental_terms_tot_2'].sum()
########################################
phrase_patterns = [nlp.make_doc(phrase_text) for phrase_text in phrase_terms]

phrase_matcher.add("TerminologyListPhrase", None, *phrase_patterns)
regression_total_count = 0
response_text = regression_data.spelling_checked_response_text.replace(np.nan, '', regex=True)
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

#common_first_response_combined = pd.read_csv('./fasc_data/machine_scores/combined_machine_scores.csv', encoding = "ISO-8859-1")
regression_data_sample_data = regression_data
regression_data_sample_data['nlp_count'] = regression_count_new_col
regression_data_sample_data['adjective_list'] = regression_adjective_list
regression_data_sample_data['verb_list'] = regression_verb_list
regression_data_sample_data['noun_list'] = regression_noun_list
regression_data_sample_data['adverb_list'] = regression_adverb_list
regression_data_sample_data['phrase_list'] = regression_phrase_list
regression_data.to_csv('./fasc_data/machine_mental_terms_descriptives.csv', encoding='utf-8', index=False)
old_mental_terms_count = regression_data.Mental_terms_tot_2
regression_data['Mental_terms_tot_2'] = regression_count_new_col

regression_data.to_csv('./fasc_data/machine_scores_mental_terms/machine_mental_terms_score.csv', encoding='utf-8', index=False)
regression_data_other_rounds_and_asd.to_csv('./fasc_data/machine_scores_mental_terms/other_rounds.csv', encoding='utf-8', index=False)
regression_data_round_three_ASD_1.to_csv('./fasc_data/machine_scores_mental_terms/last_round.csv', encoding='utf-8', index=False)
os.chdir("./fasc_data/machine_scores_mental_terms")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], sort=False)
combined_csv.to_csv( "combined_machine_scores_mental_terms.csv", index=False, encoding='utf-8-sig')
nlp_count_arr = []
mental_terms_tot_arr = []
for row in regression_data.Mental_terms_tot_2:
  nlp_count_arr.append(int(row))

can_do_comparison = True
for row in old_mental_terms_count:
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

print(mental_terms_tot_arr, nlp_count_arr)
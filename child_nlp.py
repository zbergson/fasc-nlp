import spacy
from spacy.matcher import PhraseMatcher
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
nlp = spacy.load("en_core_web_sm")
import csv

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

# Only run nlp.make_doc to speed things up
phrase_patterns = [nlp.make_doc(phrase_text) for phrase_text in phrase_terms]

phrase_matcher.add("TerminologyListPhrase", None, *phrase_patterns)

doc = nlp(u"they are going to bully him")
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
## 
adj_intersect = set(adjective_terms).intersection(adjectives)
print(adj_intersect, 'adjective')
verb_intersect = set(verb_terms).intersection(verbs)
print(verb_intersect, 'verb')
noun_intersect = set(noun_terms).intersection(nouns)
print(noun_intersect, 'noun')
adverb_intersect = set(adverb_terms).intersection(adverbs)
print(adverb_intersect, 'adverb')

phrase_matches = phrase_matcher(doc)
for match_id, start, end in phrase_matches:
    span = doc[start:end]
    print(span.text, 'phrase')

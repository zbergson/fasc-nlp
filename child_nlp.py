import spacy
from spacy.matcher import PhraseMatcher
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)

terms = []
with open('./mental-terms.txt', 'r') as wordDoc:
    for line in wordDoc:
      currentPlace = line[:-1]
      terms.append(currentPlace)


# Only run nlp.make_doc to speed things up
patterns = [nlp.make_doc(text) for text in terms]
matcher.add("TerminologyList", None, *patterns)
doc = nlp(u"To avoid upsetting his aunt. Might be short tempered or have a bad meltdown when angry.")
matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    print(span.text)
import spacy
nlp = spacy.load('en_acl_terms_sm')

from pyresparser import ResumeParser
data = ResumeParser('test.pdf').get_extracted_data()
print()
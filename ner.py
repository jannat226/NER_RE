import json 
import scispacy

with open('newSample_preprocess.json', 'r', encoding = 'utf-8') as f:
    docs = json.load(f)
    
nlp = spacy.load("en_ner_bc5cdr_md")

for doc in docs:
    spacy_doc = nlp(doc['text'])
    entities = []
    for entity in spacy_doc.ents:
        entities.append({
            "start" : entity.start_char,
            "end" : entity.end_char,
            "text_span" : entity.text,
            "label" : entity.label_,
        })
    doc['entities'] = entities
    


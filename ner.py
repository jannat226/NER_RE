# import json 
# import scispacy
# import spacy

# with open('newSample_preprocess.json', 'r', encoding = 'utf-8') as f:
#     docs = json.load(f)
    
# nlp = spacy.load("en_ner_bc5cdr_md")

# ner_results = {}
# for doc in docs:
#     spacy_doc = nlp(doc['text'])
#     entities = []
#     for entity in spacy_doc.ents:
#         entities.append({
#             "start" : entity.start_char,
#             "end" : entity.end_char,
#             "text_span" : entity.text,
#             "label" : entity.label_,
#         })
#     ner_results[doc['doc_id']] = {
#         "text" : doc['text'],
#         "entities" : entities
#     }
    
# with open('sample_ner_results.json', 'w', encoding='utf-8') as f:
#     json.dump( ner_results, f, indent=2 )

import json
import spacy

# Load your preprocessed document list
with open('newSample_preprocess.json', 'r', encoding='utf-8') as f:
    docs = json.load(f)

# Load fine-grained biomedical NER model
nlp = spacy.load("en_ner_bionlp13cg_md")

# Optionally, list the available entity types for confirmation
print("Entity labels in model:", nlp.get_pipe('ner').labels)

# Extract entities for each doc
ner_results = {}
for doc in docs:
    spacy_doc = nlp(doc['text'])
    entities = []
    for entity in spacy_doc.ents:
        entities.append({
            "start": entity.start_char,
            "end": entity.end_char,
            "text_span": entity.text,
            "label": entity.label_,  # Should be fine-grained, e.g. ORGANISM, SIMPLE_CHEMICAL, etc.        
    })
        
    ner_results[doc['doc_id']] = {
        "text": doc['text'],
        "entities": entities
    }

# Save the results
with open('sample_ner_results_bionlp.json', 'w', encoding='utf-8') as f:
    json.dump(ner_results, f, ensure_ascii=False, indent=2)

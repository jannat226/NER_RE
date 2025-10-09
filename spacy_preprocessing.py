import pandas as pd
import re
import json
import spacy 

df = pd.read_csv('sample.csv', delimiter=',', encoding='utf-8')
nlp = spacy.load("en_ner_bc5cdr_md")

def extract_section(text, section_name):
    pattern = rf"{section_name}[\s:\-]*([\s\S]+?)(?=\n[A-Z][a-zA-Z\s0-9\(\):\-.,]*\n|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE )
    if match:
        return match.group(1).strip()
    return None


def extract_figures(text):
    return re.findall(r'Figure\s*\d+|\bFig\.?\s*\d+', text)

def extract_tables(text):
    return re.findall(r'Table\s*\d+', text)

def get_entities(text, accept_labels=None):
    doc = nlp(text if text else "")
    if accept_labels:
        return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in accept_labels]
    return [(ent.text, ent.label_) for ent in doc.ents]

docs = []

for _, row in df.iterrows():
    paper_id = row['Unnamed: 0'] if 'Unnamed: 0' in row else row[0]
    full_text = str(row['0']) if '0' in row else str(row[1])
    
    # Sectional extraction (text)
    abstract = extract_section(full_text, 'Abstract')
    introduction = extract_section(full_text, 'Introduction')
    abstract = extract_section(full_text, 'Abstract')
    introduction = extract_section(full_text, 'Introduction')
    materials = extract_section(full_text, 'Materials')
    methods = extract_section(full_text, 'Methods')
    study_design = extract_section(full_text, 'Study design')
    data_analysis = extract_section(full_text, 'Data analysis')
    results = extract_section(full_text, 'Results')
    discussion = extract_section(full_text, 'Discussion')
    acknowledgement = extract_section(full_text, 'Acknowledgement')
    references = extract_section(full_text, 'References')
        
    # Entity extraction by section using scispaCy labels
    doc = {
        'paper_id': paper_id,
        'author_entities': get_entities(full_text, accept_labels=['PERSON']),    
    }
    docs.append(doc)

with open('preprocessed_papers_entities.json', 'w', encoding='utf-8') as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("All sections and scientific entities saved to preprocessed_papers_entities.json")



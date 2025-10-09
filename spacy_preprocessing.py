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

docs = []

for _, row in df.iterrows():
    paper_id = row['Unnamed: 0'] if 'Unnamed: 0' in row else row[0]
    full_text = str(row['0']) if '0' in row else str(row[1])
    
    # Sectional extraction (text)
    abstract = extract_section(full_text, 'Abstract')
    introduction = extract_section(full_text, 'Introduction')
   
        
    # Entity extraction by section using scispaCy labels
    doc = {
        'paper_id': paper_id,
        'author_entities': get_entities(full_text, accept_labels=['PERSON']),    
    }
    docs.append(doc)

with open('preprocessed_papers_entities.json', 'w', encoding='utf-8') as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("All sections and scientific entities saved to preprocessed_papers_entities.json")



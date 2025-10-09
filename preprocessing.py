import pandas as pd
import re
import json


df = pd.read_csv('sample.csv')
print(df)


# what stuff are there in the sample data 
# 1. paper ID
# 2. Author -> name -> affiliations, year 
# 3. Materials:
# Figures
# 4. Tables
# 5. Abstracts
# 6. Introduction
# 7. Methods -> detailed biomedical content related to cancer, biology, medical research, and related studies
# 8. Study design
# 9. Data analysis and statistical methods
# 10. Results
# 11. Discussion
# 12. Acknowledgement
# 13. references

def extract_author_names(text):
    #split text into lines based on whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # 2 or 3 words with capital first letters
    name_pattern = re.compile(r'^[A-Z][a-z]+(?: [A-Z][a-z]+){1,2}$')
    blacklist = set([
        "Ministry of Education", "School of", "University", "Institute", "Department",
        "Center", "Laboratory", "Hospital", "Key Laboratory", "China"
    ])
    #containing the words and not the blacklisted words
    return [line for line in lines if name_pattern.match(line) and not any(b in line for b in blacklist)]

def extract_section(text, section_name):
   # Pull text between this section header and the next one or end of string
    pattern = rf"{section_name}[:.-]?\s*(.+?)(?=(?:\n[A-Z][a-zA-Z ]{3,}[:.-]?|\Z))"
    pattern = rf"{section_name}[\s:\-]*([\s\S]+?)(?=\n[A-Z][a-zA-Z\s0-9\(\):\-.,]*\n|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_figures(text):
    return re.findall(r'Figure\s*\d+|\bFig\.?\s*\d+', text)

def extract_tables(text):
    return re.findall(r'Table\s*\d+', text)

# Read CSV as before
df = pd.read_csv('sample.csv', delimiter=',', encoding='utf-8')

docs = []
#iterrows -> loop through each of the item in df 
for _, row in df.iterrows():
    paper_id = row['Unnamed: 0']
    text = str(row['0'])
    doc = {
        'paper_id': paper_id,
        'author_names': extract_author_names(text),
        'materials': extract_section(text, 'Materials'),
        'figures': extract_figures(text),
        'tables': extract_tables(text),
        'abstract': extract_section(text, 'Abstract'),
        'introduction': extract_section(text, 'Introduction'),
        'methods': extract_section(text, 'Methods'),
        'study_design': extract_section(text, 'Study design'),
        'data_analysis': extract_section(text, 'Data analysis'),
        'results': extract_section(text, 'Results'),
        'discussion': extract_section(text, 'Discussion'),
        'acknowledgement': extract_section(text, 'Acknowledgement'),
        'references': extract_section(text, 'References')
    }
    docs.append(doc)

with open('preprocessed_papers.json', 'w', encoding='utf-8') as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("ll sections in preprocessed_papers.json")
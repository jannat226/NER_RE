import pandas as pd
import re
import json

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


# Read the CSV
df = pd.read_csv('sample.csv', delimiter=',', encoding='utf-8')

docs = []
#iterrows -> loop through each of the item in df 
for _, row in df.iterrows():
    paper_id = row['Unnamed: 0']
    text = str(row['0'])
    doc = {
        'paper_id': paper_id,
        'author_names': extract_author_names(text)
    }
    docs.append(doc)

with open('only_author_names.json', 'w', encoding='utf-8') as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("Done! Results in only_author_names.json")

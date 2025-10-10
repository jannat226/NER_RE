import pandas as pd
import re
import json

# what stuff are there in the sample data 
# 1. paper ID
# 2.  title + abstract

df = pd.read_csv('new_sample.csv')
print(df)
print(df.columns)
df = df.rename(columns={'Unnamed: 0' : 'paper_id', '0' : 'text'})

data = []
for _, row in df.iterrows():
    data.append({'paper_id': str(row['paper_id']), 'text': row['text']})


with open('newSample_preprocess.json','w', encoding = 'utf-8') as f:
    json.dump(data, f, ensure_ascii = False, indent = 2)




from llama_index.llms.ollama import Ollama 
from llama_index.core.llms import ChatMessage
import re
from typing import List
import json

with open('newSample_preprocess.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

generation_model = Ollama(
    model="qwen3:14b",
    request_timeout=300,
    context_window=8000,
)

all_entities = {}



def extract_answer_after_think(title_abstract):
    # remove  <think> ... </think> 
    if '</think>' in title_abstract:
        split_title_abstract = title_abstract.split('</think>')
        return split_title_abstract[-1].strip()
    return title_abstract.strip()

for paper in papers:
    query = paper['title_abstract']
    if not query.strip():
        print("Warning: Empty title_abstract for paper_id:", paper['paper_id'])
        
    prompt = (
        "Extract all named entities from the following text. "
        "For each entity, return a JSON object in this format:\n"
        "{\n"
        "  \"start\": <start_char index>,\n"
        "  \"end\": <end_char index>,\n"
        "  \"text_span\": <entity string from text>,\n"
        "  \"label\": <entity type>\n"
        "}\n"
        "Return a JSON list of such objects. Do not add any explanation or extra text.\n\n"
        f"Text:\n{query}"
    )
    
    messages = [
        ChatMessage(role="system", content="You are a named entity recognition (NER) system. Respond only with a JSON list as told."),
        ChatMessage(role="user", content=prompt),
    ]

    pred_text = generation_model.chat(messages)
    raw_response = pred_text.message.content
    answer_text = extract_answer_after_think(raw_response)
    print("Debug answer text:", repr(answer_text))
    try:
        entities = json.loads(answer_text)  
        all_entities[paper['paper_id']] = {
            "title_abstract": query,
            "entities": entities
        }

    except Exception as e:
        print(f"JSON parse error: {e}, original text: {repr(answer_text)}")
        all_entities[paper['paper_id']] = []  
        
    print("Paper ID:", paper['paper_id'])
    print("the entities are ",entities)
    
with open('ner_entities_using_llm.json', 'w', encoding='utf-8') as f:
    json.dump(all_entities, f, ensure_ascii=False, indent=2)

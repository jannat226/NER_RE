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
        print("Warning: Empty text for paper_id:", paper['paper_id'])
        
    prompt = (
        "Extract all named entities from the following text. "
        "For each entity, provide a dictionary with the following keys: "
        "\"start\" (character index where the entity begins), "
        "\"end\" (character index where the entity ends), "
        "\"text_span\" (the substring for the entity), and "
        "\"label\" (the entity type)."
        "\nOutput a JSON list ONLY. Do NOT include explanations, headers, 'entities:', or any extra text."
        "\nExample:\n"
        "[\n"
        "  {\"start\": 16, \"end\": 39, \"text_span\": \"color-tunable ultralong\", \"label\": \"SIMPLE_CHEMICAL\"},\n"
        "  {\"start\": 613, \"end\": 616, \"text_span\": \"ACl\", \"label\": \"GENE_OR_GENE_PRODUCT\"},\n"
        "  {\"start\": 1137, \"end\": 1140, \"text_span\": \"ISC\", \"label\": \"CELL\"}\n"
        "]\n"
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
    # all_entities[paper['paper_id']] = entities
    
with open('ner_entities_using_fewShots_llm.json', 'w', encoding='utf-8') as f:
    json.dump(all_entities, f, ensure_ascii=False, indent=2)

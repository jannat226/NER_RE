from llama_index.llms.ollama import Ollama 
from llama_index.core.llms import ChatMessage
import re
from typing import List
import json

with open('ner_entities_fewshot_llm.json', 'r', encoding='utf-8') as f:
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

for paper_id, paper_data in papers.items():
    query = paper_data['title_abstract']
    if not query.strip():
        print("Warning: Empty text for paper_id:", paper_data['paper_id'])
        
    prompt = (
        ""
        "Extract the relationship between the entities, the entities are identified from the abstracts of research papers, using the text from the abstract find the relationship between them"
        f'Given the following title and abstract:\n\n'
        f'"{paper_data["title_abstract"]}"\n\n'
        f"The identified entities are:\n"
        + ", ".join([f'"{e["text_span"]}" [{e["label"]}]' for e in paper_data["entities"]]) +
        "\n\nExtract relationships between these entities from the text."
        "Return ONLY a JSON object with this format:"
        "{\n"
        '  "title_abstract": "<title and abstract text>",\n'
        '  "entities": [\n'
        '    {\n'
        '      "entity": "<entity name>",\n'
        '      "label": "<entity label>"\n'
        '    }\n'
        '  ],\n'
        '  "relations": [\n'
        '    {\n'
        '      "head": "<head entity name>",\n'
        '      "head_type": "<head entity label>",\n'
        '      "relation": "<relationship>",\n'
        '      "tail": "<tail entity name>",\n'
        '      "tail_type": "<tail entity label>"\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "Do not include any explanations or extra text."
    )

    

    
    messages = [
        ChatMessage(role="system", content="You are a relationship extraction (RE) system. Respond only with a JSON list as told."),
        ChatMessage(role="user", content=prompt),
    ]

    pred_text = generation_model.chat(messages)
    raw_response = pred_text.message.content
    answer_text = extract_answer_after_think(raw_response)
    print("Debug answer text:", repr(answer_text))
    try:
        entities = json.loads(answer_text) 
        all_entities[paper_id] = {
            "title_abstract": query,
            "entities": entities
        }
    except Exception as e:
        print(f"JSON parse error: {e}, original text: {repr(answer_text)}")
        all_entities[paper_id] = [] 
        
    print("Paper ID:", {paper_id})
    print("the entities are ",entities)
    # all_entities[paper['paper_id']] = entities
    
with open('re_using_qwen.json', 'w', encoding='utf-8') as f:
    json.dump(all_entities, f, ensure_ascii=False, indent=2)

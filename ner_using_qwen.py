from llama_index.llms.ollama import Ollama 
from llama_index.core.llms import ChatMessage
import json

with open('newSample_preprocess.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

generation_model = Ollama(
    model="qwen3:14b",
    request_timeout=300,
    context_window=8000,
)

all_entities = {}
for paper in papers:
    query = paper['text']
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
    try:
        entities = json.loads(pred_text.message.content)
    except Exception:
        print(f"Failed to parse JSON for paper_id {paper['paper_id']}:")
        print(pred_text.message.content)
        entities = []
    all_entities[paper['paper_id']] = entities

with open('ner_entities_using_llm.json', 'w', encoding='utf-8') as f:
    json.dump(all_entities, f, ensure_ascii=False, indent=2)

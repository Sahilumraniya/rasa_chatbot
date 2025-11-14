import json
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load local models once
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
gen_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
gen_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Cache JSON data
DATA_PATH = "company_data.json"
with open(DATA_PATH, "r") as f:
    COMPANY_DATA = json.load(f)

def flatten_json(obj, parent_key="", sep="."):
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}[{i}]"
            items.extend(flatten_json(v, new_key, sep=sep))
    else:
        items.append((parent_key, str(obj)))
    return items

def infer_reasoning(query):
    """Finds the most relevant piece of info from JSON and generates a natural answer."""
    flat_data = flatten_json(COMPANY_DATA)
    facts = [f"{k.replace('.', ' ')} is {v}" for k, v in flat_data]

    # Encode query and all facts
    q_emb = embed_model.encode(query, convert_to_tensor=True)
    f_embs = embed_model.encode(facts, convert_to_tensor=True)

    # Compute similarity
    cos_scores = util.cos_sim(q_emb, f_embs)[0]
    best_idx = int(torch.argmax(cos_scores))
    best_fact = facts[best_idx]
    best_score = float(cos_scores[best_idx])

    # If not confident, respond gracefully
    if best_score < 0.55:
        return "I’m not confident about that yet."

    # Generate natural response using small local LLM
    prompt = f"Q: {query}\nA based on company info: {best_fact}\nA:"
    input_ids = gen_tokenizer.encode(prompt, return_tensors="pt")
    output = gen_model.generate(
        input_ids,
        max_length=80,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.3
    )
    text = gen_tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean up answer
    text = text.split("A:")[-1].strip()
    return text

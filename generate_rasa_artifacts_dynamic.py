# generate_rasa_artifacts_dynamic.py
import os, yaml, json
from pathlib import Path

def dump_yaml(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

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

def generate_sentences(flat_data, company_name="the company"):
    facts = []
    for key, value in flat_data:
        key_display = key.replace(".", " ").replace("_", " ")
        value_str = value.strip()
        facts.append(f"{key_display} is {value_str}.")
    return facts

def infer_intents_from_facts(facts):
    intents = {"ask_info": [], "smalltalk": []}
    question_words = ["who", "what", "when", "where", "how", "tell me about", "describe"]
    for f in facts[:200]:
        subj = f.split(" is ")[0]
        for q in question_words:
            intents["ask_info"].append(f"- {q} {subj}?")
    intents["smalltalk"].extend(["- hi", "- hello", "- hey"])
    return intents

def generate_nlu_from_json(data):
    flat = flatten_json(data)
    facts = generate_sentences(flat, data.get("company_name", "the company"))
    intents = infer_intents_from_facts(facts)
    nlu = {"version":"3.1","nlu":[]}
    for intent, examples in intents.items():
        nlu["nlu"].append({"intent": intent, "examples": "\n".join(examples)})
    return nlu

def generate_domain_dynamic(data):
    return {
        "version": "3.1",
        "intents": ["ask_info", "smalltalk"],
        "responses": {
            "utter_greet": [{"text": "Hey 👋! I'm your AI assistant. What can I help with?"}],
            "utter_default": [{"text": "Sorry, I don't have that information yet."}]
        },
        "actions": ["action_get_info"]
    }

def generate_stories_dynamic():
    return {
        "version": "3.1",
        "stories": [
            {"story": "ask info", "steps": [{"intent":"ask_info"},{"action":"action_get_info"}]},
            {"story": "smalltalk", "steps": [{"intent":"smalltalk"},{"action":"utter_greet"}]}
        ]
    }

def generate_rules_dynamic():
    return {
        "version":"3.1",
        "rules":[
            {"rule":"rule_info","steps":[{"intent":"ask_info"},{"action":"action_get_info"}]},
            {"rule":"rule_smalltalk","steps":[{"intent":"smalltalk"},{"action":"utter_greet"}]}
        ]
    }

def generate_all_dynamic(json_path, out_dir="rasa_dynamic"):
    with open(json_path,"r",encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    dump_yaml(generate_nlu_from_json(data), os.path.join(out_dir,"data","nlu.yml"))
    dump_yaml(generate_domain_dynamic(data), os.path.join(out_dir,"domain.yml"))
    dump_yaml(generate_stories_dynamic(), os.path.join(out_dir,"data","stories.yml"))
    dump_yaml(generate_rules_dynamic(), os.path.join(out_dir,"data","rules.yml"))
    print("✅ Generated in", out_dir)

if __name__=="__main__":
    generate_all_dynamic("company_data.json", out_dir="rasa_dynamic")

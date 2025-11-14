import json
from pathlib import Path

def append_qa_to_store(q, a, path="./actions/data/qa_store.json"):
    p = Path(path)
    data = []
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("⚠️ Corrupted QA store, resetting it.")
    # avoid duplicates
    if not any(d["q"].strip().lower() == q.strip().lower() for d in data):
        data.append({"q": q, "a": a})
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"✅ Added new QA: {q[:60]}...")
        return True
    print("ℹ️ Skipped duplicate QA.")
    return False

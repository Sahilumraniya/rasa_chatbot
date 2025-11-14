import os, subprocess, json, asyncio, inspect
from rasa.core.agent import Agent
from generate_rasa_artifacts_dynamic import generate_all_dynamic

# STEP 1: Generate Rasa Files
json_path = "company_data.json"
out_dir = "rasa_dynamic"

print("🧠 Generating Rasa files dynamically...")
generate_all_dynamic(json_path, out_dir=out_dir)

# STEP 2: Train Model
print("⚙️ Training Rasa model...")
subprocess.run([
    "rasa", "train",
    "--domain", "rasa_dynamic/domain.yml",
    "--data", "rasa_dynamic/data"
], check=True)
print("✅ Model training complete!\n")

# STEP 3: Load model and chat
model_dir = "models"
latest_model = max(
    [os.path.join(model_dir, f) for f in os.listdir(model_dir)],
    key=os.path.getctime
)

print(f"📦 Loading model: {latest_model}")

# ---- FIX HERE ----
agent_load = Agent.load(latest_model)
# if inspect.isawaitable(agent_load):
#     agent = asyncio.get_event_loop().run_until_complete(agent_load)
# else:
#     agent = agent_load
# ------------------

print("💬 Chatbot ready! Type your questions (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    handle = agent_load.handle_text(user_input)
    # if inspect.isawaitable(handle):
    #     responses = asyncio.get_event_loop().run_until_complete(handle)
    # else:
    #     responses = handle

    if handle:
        texts = []
        for r in handle:
            if isinstance(r, dict) and r.get("text"):
                texts.append(r["text"])
            elif hasattr(r, "get") and r.get("text"):
                texts.append(r.get("text"))
        print("Bot:", " ".join(texts) if texts else handle)
    else:
        print("Bot: Hmm, I don’t have that info yet.")

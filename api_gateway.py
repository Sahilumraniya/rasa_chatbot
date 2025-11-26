import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Any
import httpx
from better_profanity import profanity 

class KBItem(BaseModel):
    question: str
    answer: str
    embedding: List[float]

class InputPayload(BaseModel):
    message_id: str
    message: str
    user_group: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    knowledge_base: Optional[List[KBItem]] = None

class OutputPayload(BaseModel):
    user_group: str
    text: str
    message_id: str

app = FastAPI()

# URL for your Rasa server (run with: rasa run)
RASA_WEBHOOK_URL = "http://localhost:5005/webhooks/rest/webhook"

# Initialize profanity filter once
profanity.load_censor_words()

@app.post("/chat")
async def handle_chat(payload: InputPayload) -> OutputPayload:
    """
    1. Receive your custom payload.
    2. Check for profanity in payload.message.
       - If vulgar -> return "please rephrase" WITHOUT calling Rasa.
    3. If clean -> call Rasa with the same 'message' field.
    4. Return Rasa's first text response in your custom shape.
    """

    user_text = (payload.message or "").strip()

    # --- 1) Profanity detection at gateway ---
    if user_text and profanity.contains_profanity(user_text):
        return OutputPayload(
            user_group=payload.user_group,
            text="I can't respond to messages with offensive language. Please rephrase without profanity.",
            message_id=payload.message_id,
        )

    # --- 2) Build payload for Rasa (KEEPING 'message' like your original) ---
    rasa_payload = {
        "sender": payload.user_group,
        "message": user_text,
        "metadata": {
            "user_id": payload.user_id,
            "agent_id": payload.agent_id,
            "knowledge_base": [item.dict() for item in payload.knowledge_base] if payload.knowledge_base else [],
        },
    }

    # --- 3) Call Rasa Server ---
    rasa_response_text = "Sorry, I couldn't connect to the bot."
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RASA_WEBHOOK_URL,
                json=rasa_payload,
                timeout=10.0,
            )
            response.raise_for_status()

            # Extract first text response from Rasa
            rasa_data = response.json()
            if rasa_data and isinstance(rasa_data, list):
                rasa_response_text = rasa_data[0].get("text") or rasa_response_text

    except httpx.HTTPStatusError as e:
        rasa_response_text = f"Error: Could not reach Rasa server. {e}"
        print(f"Rasa server returned an error: {e}")
    except httpx.RequestError as e:
        rasa_response_text = f"Error: Failed to connect to Rasa server. {e}"
        print(f"Failed to connect to Rasa server: {e}")
    except Exception as e:
        rasa_response_text = "An unexpected error occurred."
        print(f"An unexpected error: {e}")

    # --- 4) Transform for Output ---
    return OutputPayload(
        user_group=payload.user_group,
        text=rasa_response_text,
        message_id=payload.message_id,
    )

if __name__ == "__main__":
    print("Starting API Gateway on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

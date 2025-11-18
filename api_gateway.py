import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Any
import httpx

class KBItem(BaseModel):
    question: str
    answer: str
    embedding: List[float]

class InputPayload(BaseModel):
    message_id: str
    message: str
    user_group: str
    knowledge_base: List[KBItem]

class OutputPayload(BaseModel):
    user_group: str
    text: str
    message_id: str

app = FastAPI()

# This is the URL for your Rasa server (which you run with 'rasa run')
RASA_WEBHOOK_URL = "http://localhost:5005/webhooks/rest/webhook"

@app.post("/chat")
async def handle_chat(payload: InputPayload) -> OutputPayload:
    """
    This is your new API endpoint.
    1. It receives your custom payload.
    2. It transforms it into a Rasa payload (with metadata).
    3. It calls the Rasa server.
    4. It transforms the Rasa response back into your custom format.
    """
    
    # --- 1. Transform for Rasa ---
    # We use 'user_group' as the session ID (sender)
    # We pass the 'knowledge_base' into the metadata
    rasa_payload = {
        "sender": payload.user_group,
        "message": payload.message,
        "metadata": {
            "knowledge_base": [item.dict() for item in payload.knowledge_base]
        }
    }

    # --- 2. Call Rasa Server ---
    rasa_response_text = "Sorry, I couldn't connect to the bot."
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(RASA_WEBHOOK_URL, json=rasa_payload, timeout=10.0)
            response.raise_for_status() # Raise an error if the request failed
            
            # Extract the first text response from Rasa
            rasa_data = response.json()
            if rasa_data and isinstance(rasa_data, list):
                rasa_response_text = rasa_data[0].get("text")
            
    except httpx.HTTPStatusError as e:
        rasa_response_text = f"Error: Could not reach Rasa server. {e}"
        print(f"Rasa server returned an error: {e}")
    except httpx.RequestError as e:
        rasa_response_text = f"Error: Failed to connect to Rasa server. {e}"
        print(f"Failed to connect to Rasa server: {e}")
    except Exception as e:
        rasa_response_text = "An unexpected error occurred."
        print(f"An unexpected error: {e}")

    # --- 3. Transform for Output ---
    output = OutputPayload(
        user_group=payload.user_group,
        text=rasa_response_text,
        message_id=payload.message_id
    )
    
    return output

if __name__ == "__main__":
    print("Starting API Gateway on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
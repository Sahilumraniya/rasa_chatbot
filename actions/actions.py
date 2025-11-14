import re
import logging
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

from actions.vector_store import VectorRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------
# Global Retriever (loads once from API)
# ---------------------------------------
RETRIEVER = VectorRetriever(
    api_url="http://localhost:4040/v1/vector?$limit=-1",
)

class ActionRememberName(Action):
    def name(self) -> str:
        return "action_remember_name"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        entities = tracker.latest_message.get("entities", [])
        user_name = next((e.get("value") for e in entities if e.get("entity") == "user_name"), None)

        if not user_name:
            # try to guess name from sentence
            text = tracker.latest_message.get("text", "")
            tokens = [t for t in text.split() if t.lower() not in ("i", "am", "my", "name", "is", "hello", "hi", "hey", "this")]
            if tokens:
                user_name = tokens[-1]

        if user_name:
            dispatcher.utter_message(text=f"Nice to meet you, {user_name} 😊")
            return [SlotSet("user_name", user_name)]
        else:
            dispatcher.utter_message(text="I didn't catch your name — can you tell me again?")
            return []


# --------------------------
# Action: Vector Reasoner (NO LLM)
# --------------------------
# class ActionVectorReasoner(Action):
#     def name(self) -> str:
#         return "action_vector_reasoner"

#     async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
#         query = tracker.latest_message.get("text", "").strip()
#         if not query:
#             dispatcher.utter_message(text="What would you like to know?")
#             return []

#         # --- Step 1: Vector Search (top 5) ---
#         results = RETRIEVER.search(query, top_k=5)
#         logger.info(f"[Vector Search] Query: {query} | Top {len(results)} matches")

#         if not results:
#             dispatcher.utter_message(text="I don’t have information about that yet.")
#             return []

#         # --- Step 2: Clean & Prepare Outputs ---
#         formatted_chunks = []
#         for text, score in results:
#             clean = text
#             clean = re.sub(r"Q:\s*", "", clean)
#             clean = re.sub(r"A:\s*", "", clean)
#             clean = re.sub(r"\s{2,}", " ", clean).strip()
#             formatted_chunks.append(clean)

#         # Debug print
#         print("----- VECTOR MATCHES -----", flush=True)
#         for chunk in formatted_chunks:
#             print(chunk, flush=True)

#         # --- Step 3: Respond Directly (NO LLM) ---
#         # Return the **top matched answer**
#         # best_answer = formatted_chunks[0]

#         best_text = formatted_chunks[0]

#         # Extract only the answer part
#         if "A:" in best_text:
#             answer = best_text.split("A:", 1)[1].strip()
#         else:
#             # answer = best_text  # fallback
#             if "?" in best_text:
#                 # If there's a question mark, assume it's a Q&A format
# #                 What is Smartters?
# # Smartters is a software development company that specializes in helping startups and businesses build innovative products. They transform ideas into smart software solutions, offering services ranging from mobile app development to artificial intelligence.
#                 question_part, answer_part = best_text.split("?", 1)
#                 answer = answer_part.strip()
#             else:
#                 answer = best_text  # fallback

#         dispatcher.utter_message(text=answer)


#         # dispatcher.utter_message(text=best_answer)
#         return []

class ActionVectorReasoner(Action):
    def name(self): return "action_vector_reasoner"

    async def run(self, dispatcher, tracker, domain):

        query = tracker.latest_message.get("text", "").strip()

        results = RETRIEVER.search(query, top_k=1)

        if not results:
            dispatcher.utter_message("I don’t have information about that yet.")
            return []

        answer, score = results[0]
        dispatcher.utter_message(answer)

        return []



class ActionLLMFallback(Action):
    def name(self) -> str:
        return "action_llm_fallback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        dispatcher.utter_message(text="Sorry, I don't have enough information about that.")
        return []

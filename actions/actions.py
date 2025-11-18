import re
import logging
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from typing import Any, Text, Dict, List
from rasa_sdk import logger

from rasa_sdk.forms import FormValidationAction

try:
    from actions.vector_store import VectorRetriever 
    RETRIEVER = VectorRetriever()
    logger.info("Successfully connected to VectorRetriever API.")
except Exception as e:
    logger.error(f"Failed to initialize VectorRetriever: {e}")
    RETRIEVER = None

logger = logging.getLogger(__name__)


class ValidateNameForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_name_form"

    def validate_user_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate 'user_name' value."""


        name = slot_value
        name = re.sub(r"it's|it is|my name is|I am|I'm|people call me|you can call me|this is", "", name, flags=re.IGNORECASE)
        name = name.strip().split(" ")[0]
        name = name.title()

        if not name:
            dispatcher.utter_message(text="I'm sorry, I didn't catch that. What's your name?")
            return {"user_name": None}
        
        logger.info(f"Successfully extracted name: {name}")
        return {"user_name": name}

class ActionVectorReasoner(Action):
    def name(self): 
        return "action_vector_reasoner"
    
    async def run(self, dispatcher, tracker, domain):
        
        if not RETRIEVER:
            dispatcher.utter_message("My knowledge base is currently offline. Please try again later.")
            return []

        query = tracker.latest_message.get("text", "").strip()
        metadata = tracker.latest_message.get("metadata", {})
        knowledge_base = metadata.get("knowledge_base") or []

        # print("knowledge_base:", knowledge_base)
        
        try:
            RETRIEVER.refresh(knowledge_base) 
            results = RETRIEVER.search(query, top_k=1)
            print("Vector search results:", results)
            
            if not results:
                dispatcher.utter_message("I don’t have information about that yet.")
                return []
            
            answer, score = results[0]
            dispatcher.utter_message(answer)

        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            dispatcher.utter_message("Sorry, I ran into a problem trying to find an answer.")

        return []
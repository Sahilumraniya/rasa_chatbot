import json
import re
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# We don't need Mongo for this payload-driven logic
# MONGO_URI = "mongodb+srv://soumyasmartters:EAq2rKNP01KyTCAt@automation.c59mat7.mongodb.net/form_dev"
# client = MongoClient(MONGO_URI)
# db = client.get_database("form_dev")


class VectorRetriever:
    """
    ----------------------------------------
    - THIS IS THE "POWERFUL" VERSION YOU ASKED FOR
    - It uses its OWN AI model for everything
    - It IGNORES the 'embedding' field from the payload
    - This guarantees query and docs are in the same vector space
    - It clears its state on every request
    """

    def __init__(self,
                 model_name="multi-qa-MiniLM-L6-cos-v1",
                 min_confidence=0.75):

        # This is the "single object" AI brain
        self.model = SentenceTransformer(model_name)
        self.min_confidence = min_confidence

        # State (will be rebuilt on every 'refresh' call)
        self.texts = []
        self.questions = []
        self.answers = []
        self.embeddings = None
        self.index = None
        self.use_faiss = False

    def _load_from_payload(self, payload):
        """
        Loads ONLY THE TEXT from the payload.
        It IGNORES the pre-computed 'embedding' field.
        """
        for item in payload:
            q = item.get("question")
            a = item.get("answer")
            
            # We skip this item if it has no text
            if not q or not a:
                continue

            self.questions.append(q)
            self.answers.append(a)
            self.texts.append(f"Q: {q}\nA: {a}")
        
        # We return the TEXT, not the embeddings
        return self.texts

    def _build_index(self):
        """
        Builds the FAISS index from the text embeddings
        that we compute ourselves.
        """
        if not self.embeddings.any():
            return

        # L2 normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-12)

        dim = self.embeddings.shape[1]

        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(dim) # Inner Product == Cosine Similarity
            index.add(self.embeddings.astype(np.float32))
            self.index = index
            self.use_faiss = True
            print("⚡ Using FAISS cosine similarity index")
        else:
            self.index = NearestNeighbors(
                n_neighbors=min(10, len(self.embeddings)),
                metric="cosine"
            )
            self.index.fit(self.embeddings)
            print("⚡ Using sklearn cosine similarity index")

    def refresh(self, payload=None):
        """
        This is the main function called by actions.py.
        It rebuilds the entire index from scratch
        using the provided payload.
        """
        # --- THIS IS THE FIX for the "state" bug ---
        # Clear the state from the previous request
        self.texts = []
        self.questions = []
        self.answers = []
        self.embeddings = None
        self.index = None
        # --- END OF FIX ---

        if not payload:
            print("❌ No payload provided to refresh.")
            return

        # 1. Load TEXT from the payload
        text_data = self._load_from_payload(payload)

        if not text_data:
            print("❌ No text data found in payload.")
            return

        # 2. COMPUTE NEW EMBEDDINGS using our own model
        print(f"📦 Computing embeddings for {len(text_data)} items...")
        self.embeddings = self.model.encode(text_data, convert_to_numpy=True)
        print(f"📦 Embeddings computed.")

        # 3. Build the index from our new embeddings
        self._build_index()

    def _semantic_expand(self, query):
        q = query.lower().strip()
        expansions = [q]
        words = q.split()
        if len(words) > 1:
            expansions.append(" ".join(words[::-1]))
            expansions.append(" ".join(sorted(words)))
        expansions.append(f"information about {q}")
        expansions.append(f"details of {q}")
        return list(set(expansions))


    def _search_vectors(self, emb, k=8):
        if self.use_faiss:
            q = emb.astype("float32").reshape(1, -1)
            faiss.normalize_L2(q) # Normalize the query
            D, I = self.index.search(q, k)
            return [(i, float(D[0][r])) for r, i in enumerate(I[0])]
        else:
            distances, indices = self.index.kneighbors(emb.reshape(1, -1), n_neighbors=k)
            return [(i, float(1 - dist)) for dist, i in zip(distances[0], indices[0])]

    def search(self, query, top_k=3):
        if self.embeddings is None or not self.embeddings.any():
            print("❌ Search failed: No embeddings are loaded.")
            return []

        query_clean = query.strip()
        exp_queries = self._semantic_expand(query_clean)

        scored = {}

        for q in exp_queries:
            # 1. Compute query embedding with our model
            q_emb = self.model.encode([q], convert_to_numpy=True)[0]
            # 2. Normalize it (because our index is normalized)
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

            results = self._search_vectors(q_emb, k=10)

            for idx, score in results:
                if idx not in scored:
                    scored[idx] = 0
                scored[idx] = max(scored[idx], score)

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        
        # 3. Filter by confidence
        # The scores will now be high (e.g., 0.8, 0.9)
        final = [(self.answers[i], s) for i, s in ranked if s >= self.min_confidence]
        
        if not final:
            print(f"⚠️ No results found above {self.min_confidence} threshold.")
        
        return final[:top_k]
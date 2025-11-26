import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv
load_dotenv()

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class VectorRetriever:
    def __init__(self,
                 mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
                 db_name=os.getenv("DB_NAME", "replyme_dev"),
                 collection_name="embeddedknowledgebases",
                 model_name="all-MiniLM-L6-v2",
                 min_confidence=0.75):
        
        # --- MongoDB Setup ---
        print("⏳ Connecting to MongoDB...")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        print("✅ MongoDB Connected.")

        # --- AI Model Setup ---
        # We still need the model to encode the USER QUERY during search
        print(f"⏳ Loading AI Model ({model_name})...")
        self.model = SentenceTransformer(model_name)
        self.min_confidence = min_confidence
        # print("✅ AI Model Loaded.")

        # --- Internal State ---
        self.answers = []
        self.index_to_answer_id = [] 
        self.embeddings = None
        self.index = None
        self.use_faiss = False

    def refresh(self, user_id, agent_id=None):
        """
        1. Fetches data from MongoDB.
        2. Extracts PRE-CALCULATED embeddings (question_embedding & embedding).
        3. Builds the FAISS/Sklearn index directly from stored vectors.
        """
        # Reset state
        self.answers = []
        self.index_to_answer_id = []
        self.embeddings = None
        self.index = None

        # print(f"🔄 Fetching data for User: {user_id}, Agent: {agent_id}...")

        user_id = ObjectId(user_id)
        if agent_id:
            agent_id = ObjectId(agent_id)
        else:
            agent_id = None
        
        # print(f"📥 Querying MongoDB for relevant documents for user_id: {user_id}, agent_id: {agent_id}...")

        # Filter logic as requested
        query_filter = {
            "user": user_id, 
            "status": "ACTIVE", 
            '$or': [{'agentId': agent_id}, {'agentId': None}] 
        }
        
        cursor = self.collection.find(query_filter)
        documents = list(cursor)

        # print(f"✅ Fetched {len(documents)} documents from MongoDB.")
        
        if not documents:
            print("⚠️ No data found in MongoDB for this user/agent.")
            return

        # --- 2. Extract Existing Vectors ---
        vectors_list = []
        
        # print(f"📊 Found {len(documents)} documents. Loading stored embeddings...")

        for i, doc in enumerate(documents):
            # Get text
            a_text = doc.get("answer")
            if not a_text: continue

            # Get stored vectors (List[float])
            # "question_embedding" usually maps strictly to the question
            # "embedding" usually maps to the combined Q+A or context
            q_vec = doc.get("question_embedding") 
            qa_vec = doc.get("embedding")

            # If neither exists, skip this doc (or fallback to text encoding if you wanted)
            if not q_vec and not qa_vec:
                continue

            # Store the answer text reference
            self.answers.append(a_text)

            # Strategy: Add whichever vectors exist to the index
            
            # 1. Add Question Vector (Excellent for direct question matching)
            if q_vec and isinstance(q_vec, list):
                vectors_list.append(q_vec)
                self.index_to_answer_id.append(i)

            # 2. Add Context/Combined Vector (Good for broader semantic matching)
            if qa_vec and isinstance(qa_vec, list):
                vectors_list.append(qa_vec)
                self.index_to_answer_id.append(i)

        if not vectors_list:
            print("❌ No vector data found in documents.")
            return

        # --- 3. Convert to Numpy ---
        # We convert the list of lists directly to a numpy array
        self.embeddings = np.array(vectors_list, dtype='float32')
        
        # --- 4. Normalize ---
        # Even if stored embeddings are normalized, it is safer to re-normalize 
        # locally to ensure Dot Product (Cosine Similarity) works perfectly.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-12)

        # --- 5. Build Index ---
        if FAISS_AVAILABLE:
            dim = self.embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self.embeddings)
            self.index = index
            self.use_faiss = True
            print(f"⚡ Built FAISS index with {len(self.embeddings)} vectors.")
        else:
            self.index = NearestNeighbors(n_neighbors=min(10, len(self.embeddings)), metric="cosine")
            self.index.fit(self.embeddings)
            print("⚡ Built Sklearn index.")

    def refreshKnowledgeBase(self, payload=None):
            """
            1. Reads TEXT from payload (Question and Answer).
            2. IGNORES the mismatched embeddings in the payload.
            3. Generates NEW, perfect embeddings using the local model.
            """
            self.answers = []
            self.index_to_answer_id = []
            self.embeddings = None
            self.index = None

            if not payload: return

            # --- 1. Extract Text Only ---
            text_to_embed = []
            
            print("🔄 Re-calculating embeddings from text to GUARANTEE match...")

            for i, item in enumerate(payload):
                q = item.get("question")
                a = item.get("answer")
                if not a: continue

                # Store the answer text
                self.answers.append(a)

                # Strategy: We will embed the Question AND the Answer separately
                # This maintains your "Hybrid Search" logic
                
                # Add Question text to be embedded
                if q:
                    text_to_embed.append(q)
                    self.index_to_answer_id.append(i)

                # Add Answer text to be embedded
                if a:
                    text_to_embed.append(a)
                    self.index_to_answer_id.append(i)

            if not text_to_embed:
                print("❌ No text found to embed.")
                return

            # --- 2. Generate Embeddings Locally (The Fix) ---
            # This creates vectors that are 100% compatible with the query
            # This converts the list of strings into a numpy array of vectors
            self.embeddings = self.model.encode(text_to_embed, convert_to_numpy=True)
            
            # --- 3. Normalize ---
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / (norms + 1e-12)

            # --- 4. Build Index ---
            if FAISS_AVAILABLE:
                dim = self.embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(self.embeddings.astype(np.float32))
                self.index = index
                self.use_faiss = True
                print(f"⚡ Built FAISS index with {len(self.embeddings)} vectors.")
            else:
                self.index = NearestNeighbors(n_neighbors=min(10, len(self.embeddings)), metric="cosine")
                self.index.fit(self.embeddings)
                print("⚡ Built Sklearn index.")

    def _semantic_expand(self, query):
        q = query.lower().strip()
        expansions = [q]
        words = q.split()
        if len(words) > 1:
            expansions.append(" ".join(words[::-1]))
        return list(set(expansions))

    def _search_vectors(self, emb, k=10):
        q = emb.astype("float32").reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0: q = q / norm

        if self.use_faiss:
            D, I = self.index.search(q, k)
            return [(int(I[0][r]), float(D[0][r])) for r in range(k) if I[0][r] >= 0]
        else:
            distances, indices = self.index.kneighbors(q, n_neighbors=k)
            return [(int(indices[0][r]), float(1 - distances[0][r])) for r in range(k)]

    def search(self, query, top_k=1):
        if self.embeddings is None: 
            print("⚠️ Index not built. Call refresh(user_id, agent_id) first.")
            return []

        query_clean = query.strip()
        
        # We still use the model here to encode the INCOMING QUERY
        # The query must be embedded into the same space as the stored vectors
        q_emb = self.model.encode([query_clean], convert_to_numpy=True)[0]
        
        results = self._search_vectors(q_emb, k=10)
        
        best_scores = {}
        for vec_idx, score in results:
            if vec_idx < len(self.index_to_answer_id):
                ans_id = self.index_to_answer_id[vec_idx]
                if ans_id not in best_scores: best_scores[ans_id] = 0
                best_scores[ans_id] = max(best_scores[ans_id], score)

        ranked = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)

        final = []
        for ans_id, score in ranked:
            if score >= self.min_confidence:
                final.append((self.answers[ans_id], score))
        
        return final[:top_k]
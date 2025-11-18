import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class VectorRetriever:
    def __init__(self,
                 model_name="all-MiniLM-L6-v2",
                 min_confidence=0.75):
        
        print(f"⏳ Loading AI Model ({model_name})...")
        # We load the model to generate ALL embeddings (Query + Docs)
        # This guarantees they match perfectly.
        self.model = SentenceTransformer(model_name)
        self.min_confidence = min_confidence
        print("✅ AI Model Loaded.")

        self.answers = []
        self.index_to_answer_id = [] 
        self.embeddings = None
        self.index = None
        self.use_faiss = False

    def refresh(self, payload=None):
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
        if self.embeddings is None: return []

        query_clean = query.strip()
        exp_queries = self._semantic_expand(query_clean)
        best_scores = {}

        for q in exp_queries:
            q_emb = self.model.encode([q], convert_to_numpy=True)[0]
            results = self._search_vectors(q_emb, k=10)

            for vec_idx, score in results:
                if vec_idx < len(self.index_to_answer_id):
                    ans_id = self.index_to_answer_id[vec_idx]
                    if ans_id not in best_scores: best_scores[ans_id] = 0
                    best_scores[ans_id] = max(best_scores[ans_id], score)

        ranked = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
        
        # if ranked:
        #     # You should see 0.99 or 1.00 here now
        #     print(f"🔍 Top Match Score: {ranked[0][1]}") 

        final = []
        for ans_id, score in ranked:
            if score >= self.min_confidence:
                final.append((self.answers[ans_id], score))
        
        return final[:top_k]
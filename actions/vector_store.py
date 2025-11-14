import json
import re
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorRetriever:
    """
    ----------------------------------------
    - Uses API embeddings
    - Normalizes vectors
    - Multi-query expansion (semantic)
    - Weighted similarity fusion
    - Softmax rank boosting
    - Extracts ONLY answer
    """

    def __init__(self,
                 api_url="http://localhost:4040/v1/vector?$limit=-1",
                 model_name="multi-qa-MiniLM-L6-cos-v1",
                 min_confidence=0.15):

        self.api_url = api_url.rstrip("/")
        self.model = SentenceTransformer(model_name)

        self.texts = []
        self.questions = []
        self.answers = []
        self.embeddings = None
        self.index = None
        self.use_faiss = False

        self.min_confidence = min_confidence
        self.refresh()

    def _load_from_api(self):
        print(f"⬇ Fetching vector data from API: {self.api_url}")

        try:
            r = requests.get(self.api_url, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print("❌ API fetch error:", e)
            return []

        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        cleaned = []

        for item in data:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except:
                    continue

            if not isinstance(item, dict):
                continue

            q = item.get("question")
            a = item.get("answer")
            emb = item.get("embedding")

            if not q or not a or not emb:
                continue

            if isinstance(emb, str):
                try:
                    emb = json.loads(emb)
                except:
                    continue

            self.questions.append(q)
            self.answers.append(a)
            self.texts.append(f"Q: {q}\nA: {a}")
            cleaned.append(emb)

        return cleaned

    def _build_index(self):
        if self.embeddings is None:
            return

        # L2 normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-12)

        dim = self.embeddings.shape[1]

        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(dim)
            index.add(self.embeddings)
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

    def refresh(self):
        raw = self._load_from_api()

        if not raw:
            print("❌ No embeddings received.")
            return

        self.embeddings = np.array(raw, dtype="float32")
        print(f"📦 Loaded {len(self.embeddings)} embeddings.")

        self._build_index()

    def _semantic_expand(self, query):
        q = query.lower().strip()

        expansions = [q]

        # Split into chunks
        words = q.split()
        if len(words) > 1:
            expansions.append(" ".join(words[::-1]))
            expansions.append(" ".join(sorted(words)))

        # Remove common filler
        expansions.append(" ".join([w for w in words if len(w) > 2]))

        # Add prefix
        expansions.append(f"information about {q}")
        expansions.append(f"details of {q}")

        return list(set(expansions))


    def _search_vectors(self, emb, k=8):
        if self.use_faiss:
            q = emb.astype("float32").reshape(1, -1)
            faiss.normalize_L2(q)
            D, I = self.index.search(q, k)
            return [(i, float(D[0][r])) for r, i in enumerate(I[0])]

        distances, indices = self.index.kneighbors(emb.reshape(1, -1), n_neighbors=k)
        return [(i, float(1 - dist)) for dist, i in zip(distances[0], indices[0])]

    def search(self, query, top_k=3):
        if not self.embeddings.any():
            return []

        query_clean = query.strip()
        exp_queries = self._semantic_expand(query_clean)

        scored = {}

        for q in exp_queries:
            q_emb = self.model.encode([q], convert_to_numpy=True)[0]
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

            results = self._search_vectors(q_emb, k=10)

            for idx, score in results:
                if idx not in scored:
                    scored[idx] = 0
                scored[idx] = max(scored[idx], score)

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        final = [(self.answers[i], s) for i, s in ranked if s >= self.min_confidence]

        return final[:top_k]

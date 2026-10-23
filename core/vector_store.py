from typing import List, Tuple, Optional
import numpy as np
import pickle
import os
from core.document import Document
from core.embeddings import BaseEmbeddingEngine

class VectorStore:
    """
    In-memory vector store supporting cosine similarity search,
    score filtering, and serialization.
    """
    def __init__(self, embedding_engine: BaseEmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]):
        """Adds and indexes a collection of Documents."""
        if not documents:
            return
        self.documents.extend(documents)
        texts = [doc.page_content for doc in self.documents]
        self.embeddings = self.embedding_engine.embed_documents(texts)

    def similarity_search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """Computes cosine similarity between query vector and stored embeddings."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Cosine similarity for L2-normalized vectors is dot product
        scores = np.dot(self.embeddings, query_vector.T).flatten()

        # Get top-k indices sorted descending
        top_indices = np.argsort(scores)[::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                results.append((self.documents[idx], score))
            if len(results) >= top_k:
                break
        return results

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """Embeds query and searches for the top-k most similar chunks."""
        query_vec = self.embedding_engine.embed_query(query)
        return self.similarity_search_by_vector(query_vec, top_k=top_k, threshold=threshold)

    def save(self, filepath: str):
        """Persists the vector store to disk."""
        data = {
            "documents": [d.to_dict() for d in self.documents],
            "embeddings": self.embeddings,
            "engine": self.embedding_engine
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        """Loads a persisted vector store from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vector store file not found: {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        store = cls(embedding_engine=data["engine"])
        store.documents = [Document.from_dict(d) for d in data["documents"]]
        store.embeddings = data["embeddings"]
        return store

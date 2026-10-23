from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class BaseEmbeddingEngine(ABC):
    """Abstract base class for RAG embedding engines."""
    @abstractmethod
    def fit(self, texts: List[str]):
        """Fit the model on corpus texts if required."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of document strings into an ndarray of shape (N, D)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into an ndarray of shape (1, D)."""
        pass

class TFIDFEmbeddingEngine(BaseEmbeddingEngine):
    """
    Fast, local, zero-dependency TF-IDF vectorizer with unigram + bigram support.
    Produces L2-normalized vectors so cosine similarity = dot product.
    """
    def __init__(self, ngram_range=(1, 2), max_features: int = 8000, sublinear_tf: bool = True):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            stop_words='english'
        )
        self.is_fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.is_fitted = True

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            self.fit(texts)
        sparse_matrix = self.vectorizer.transform(texts)
        dense_matrix = sparse_matrix.toarray()
        return normalize(dense_matrix, norm='l2', axis=1)

    def embed_query(self, query: str) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Embedding engine must be fitted before embedding queries.")
        sparse_vec = self.vectorizer.transform([query])
        dense_vec = sparse_vec.toarray()
        return normalize(dense_vec, norm='l2', axis=1)

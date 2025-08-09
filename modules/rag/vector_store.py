import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Try importing FAISS
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """
    L2 normalize a 2D NumPy array row-wise.
    Args:
        mat (np.ndarray): The matrix to normalize.
    Returns:
        np.ndarray: The L2-normalized matrix.
    """
    eps = 1e-12
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return mat / norms


class VectorStore:
    """
    Vector storage using FAISS only.
    Document metadata is stored separately in a `.meta` file.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_faiss: Optional[bool] = None):
        """
        Initialize the VectorStore.

        Args:
            model_name (str): Name of the SentenceTransformer model.
            use_faiss (bool, optional): Whether to use FAISS for indexing. Defaults to True if available.
        """
        self.documents: List[Dict[str, Any]] = []
        self.embedding_dim: Optional[int] = None
        self.model = SentenceTransformer(model_name)

        self.use_faiss: bool = _FAISS_AVAILABLE if use_faiss is None else bool(use_faiss)
        self.index = None

        if self.use_faiss and not _FAISS_AVAILABLE:
            logging.warning("FAISS requested but not available; falling back to NumPy.")
            self.use_faiss = False

        if self.use_faiss:
            logging.info("Using FAISS (IndexFlatIP) for similarity search.")
        else:
            logging.info("Using NumPy cosine similarity (no FAISS).")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add a list of documents to the vector store.

        Args:
            documents (List[Dict]): List of documents, each containing at least a "text" key.
        """
        if not documents:
            return
        texts = [doc["text"] for doc in documents]
        emb = np.asarray(self.model.encode(texts, convert_to_tensor=False), dtype=np.float32)
        emb = _l2_normalize(emb)

        if self.embedding_dim is None:
            self.embedding_dim = emb.shape[1]

        self.documents.extend(documents)

        if self.use_faiss:
            self._ensure_faiss_index()
            self.index.add(emb)
        else:
            if not hasattr(self, "_vectors"):
                self._vectors = []
            self._vectors.extend(emb)

    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for the most similar documents to the query.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict]: List of results with 'document' and 'similarity' keys.
        """
        if not self.documents:
            return []
        q = np.asarray(self.model.encode([query], convert_to_tensor=False), dtype=np.float32)
        q = _l2_normalize(q)

        if self.use_faiss and self.index is not None:
            scores, idxs = self.index.search(q, min(top_k, len(self.documents)))
            return [
                {"document": self.documents[i], "similarity": float(scores[0][n])}
                for n, i in enumerate(idxs[0]) if 0 <= i < len(self.documents)
            ]
        else:
            sims = np.dot(_l2_normalize(np.array(self._vectors)), q[0])
            top_indices = np.argsort(sims)[-top_k:][::-1]
            return [
                {"document": self.documents[i], "similarity": float(sims[i])}
                for i in top_indices
            ]

    def _ensure_faiss_index(self):
        """
        Initialize the FAISS index if it has not been created yet.
        """
        if self.index is None:
            if self.embedding_dim is None:
                raise ValueError("Cannot initialize FAISS index without embedding dimension.")
            self.index = faiss.IndexFlatIP(self.embedding_dim)

    def save(self, file_path: str) -> bool:
        """
        Save the vector store to disk.
        Produces:
            - .meta: JSON file containing document metadata
            - .faiss: FAISS index file

        Args:
            file_path (str): Path prefix for the saved files (without extension).

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Save document metadata
            with open(f"{file_path}.meta", "w", encoding="utf-8") as f_meta:
                json.dump(self.documents, f_meta, ensure_ascii=False)

            # Save FAISS index
            if self.use_faiss and self.index is not None:
                faiss.write_index(self.index, f"{file_path}.faiss")
                logging.info(f"Vector store saved: {file_path}.faiss / .meta")
            else:
                logging.warning("No FAISS index to save.")
            return True
        except Exception as e:
            logging.error(f"Error saving vector store: {e}")
            return False

    @classmethod
    def load(cls, file_path: str) -> "VectorStore":
        """
        Load a vector store from disk.

        Args:
            file_path (str): Path prefix of the saved files (without extension).

        Returns:
            VectorStore: The loaded vector store instance.
        """
        try:
            instance = cls()
            with open(f"{file_path}.meta", "r", encoding="utf-8") as f_meta:
                instance.documents = json.load(f_meta)
            if instance.documents:
                instance.embedding_dim = len(instance.model.encode([instance.documents[0]["text"]])[0])
            faiss_path = f"{file_path}.faiss"
            if instance.use_faiss and os.path.exists(faiss_path):
                instance.index = faiss.read_index(faiss_path)
                logging.info(f"Vector store loaded with FAISS index: {len(instance.documents)} docs")
            else:
                logging.warning("FAISS index not found; search will use NumPy.")
            return instance
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            raise

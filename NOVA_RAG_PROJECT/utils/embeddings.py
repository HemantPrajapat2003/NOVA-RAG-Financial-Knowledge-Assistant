"""
Embeddings utility module for Financial RAG.
Provides 384-dimensional vector embeddings using sentence-transformers/all-MiniLM-L6-v2.
Uses the high-performance local ONNX runtime implementation to guarantee 100% offline
compatibility and avoid binary DLL policy restrictions on Windows.
"""

import functools
from typing import List
from langchain_core.embeddings import Embeddings


class MiniLML6V2Embeddings(Embeddings):
    """
    LangChain compatible Embeddings wrapper for all-MiniLM-L6-v2 (384-dimensional).
    Powered by Chroma's optimized ONNX MiniLM runtime.
    """
    
    def __init__(self):
        super().__init__()
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._ef = ONNXMiniLM_L6_V2()
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.dimension = 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document chunks into 384-dimensional vectors."""
        if not texts:
            return []
        # Filter and sanitize strings
        clean_texts = [str(t) if t is not None else "" for t in texts]
        embeddings = self._ef(clean_texts)
        # Ensure return type is List[List[float]]
        return [list(map(float, vec)) for vec in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query into a 384-dimensional vector."""
        clean_text = str(text) if text is not None else ""
        embeddings = self._ef([clean_text])
        return [float(x) for x in embeddings[0]]


@functools.lru_cache(maxsize=1)
def get_embedding_function() -> Embeddings:
    """
    Factory function returning the standard all-MiniLM-L6-v2 LangChain embedding instance.
    Cached as a singleton to prevent redundant model loads.
    """
    return MiniLML6V2Embeddings()


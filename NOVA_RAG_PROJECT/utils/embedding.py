"""
Alias module for utils.embeddings for import compatibility.
Re-exports get_embedding_function and MiniLML6V2Embeddings.
"""

from utils.embeddings import get_embedding_function, MiniLML6V2Embeddings

__all__ = ["get_embedding_function", "MiniLML6V2Embeddings"]

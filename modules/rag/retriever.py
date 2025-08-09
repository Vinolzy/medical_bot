"""
Retriever Module
Responsible for retrieving relevant documents based on queries
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config
from modules.rag.vector_store import VectorStore

class Retriever:
    """
    Retriever Class, responsible for retrieving relevant documents
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        """
        Initialize retriever

        Args:
            vector_store: Vector store instance
            top_k: Number of most similar documents to return
        """
        self.vector_store = vector_store
        self.top_k = top_k
        logging.info(f"Retriever initialized with top_k: {top_k}")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the query

        Args:
            query: Query text

        Returns:
            List of relevant documents with content and similarity scores
        """
        try:
            # Use vector store for similarity search
            results = self.vector_store.similarity_search(query, self.top_k)
            logging.info(f"Query '{query}' retrieved {len(results)} relevant documents")
            return results

        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            return []

    def get_context_for_query(self, query: str) -> str:
        """
        Get context information for query

        Args:
            query: Query text

        Returns:
            Formatted context information
        """
        try:
            # Retrieve relevant documents
            results = self.retrieve(query)

            if not results:
                logging.warning(f"Query '{query}' retrieved no relevant documents")
                return ""

            # Format context
            context = "Here is information relevant to your question:\n\n"

            for i, result in enumerate(results, 1):
                document = result["document"]
                similarity = result["similarity"]

                # Add document content
                context += f"Section {i} [Relevance: {similarity:.2f}]:\n"
                context += document["text"].strip()
                context += "\n\n"

                # Add source information
                if "metadata" in document and document["metadata"]:
                    metadata = document["metadata"]
                    if "source" in metadata:
                        context += f"Source: {metadata['source']}\n\n"

            return context.strip()

        except Exception as e:
            logging.error(f"Error getting context: {e}")
            return ""
"""
RAG Manager
Unified management of Retrieval Augmented Generation (RAG) functionality
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
from modules.rag.document_processor import DocumentProcessor
from modules.rag.vector_store import VectorStore
from modules.rag.retriever import Retriever

class RAGManager:
    """
    RAG Manager Class, unified management of document processing, vector database and retriever
    """

    def __init__(self, knowledge_base_dir: Optional[str] = None):
        """
        Initialize RAG manager

        Args:
            knowledge_base_dir: Knowledge base directory path (optional)
        """
        # Set knowledge base directory
        self.knowledge_base_dir = knowledge_base_dir or os.path.join(project_root, "data", "knowledge_base")
        os.makedirs(self.knowledge_base_dir, exist_ok=True)

        # Set vector database file path
        self.vector_db_path = os.path.join(project_root, "data", "vector_store", "vector_db")
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)

        # Initialize document processor
        self.document_processor = DocumentProcessor(
            chunk_size=config.EMBEDDING_CHUNK_SIZE,
            chunk_overlap=config.EMBEDDING_CHUNK_OVERLAP
        )

        # Try to load existing vector database, create new one if not exists
        try:
            if os.path.exists(f"{self.vector_db_path}.index") and os.path.exists(f"{self.vector_db_path}.docs"):
                self.vector_store = VectorStore.load(self.vector_db_path)
                logging.info(f"Vector database loaded with {len(self.vector_store.documents)} documents")
            else:
                self.vector_store = VectorStore()
                logging.info("New vector database created")
        except Exception as e:
            logging.error(f"Error loading vector database: {e}")
            self.vector_store = VectorStore()
            logging.info("New vector database created (due to loading error)")

        # Initialize retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            top_k=3
        )

    def add_document(self, file_path: str) -> bool:
        """
        Add single document to knowledge base

        Args:
            file_path: Document path

        Returns:
            Whether document was successfully added
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                return False

            # Check file type
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in ['.pdf', '.txt', '.docx', '.doc']:
                logging.error(f"Unsupported file type: {file_ext}")
                return False

            # Prepare target file path - copy file to knowledge base directory
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(self.knowledge_base_dir, file_name)

            # Copy to knowledge base directory if source file is not already there
            if file_path != dest_path:
                import shutil
                shutil.copy2(file_path, dest_path)
                logging.info(f"File copied to knowledge base directory: {dest_path}")

            # Process document
            logging.info(f"Starting document processing: {file_path}")
            chunks = self.document_processor.process_file(file_path)
            logging.info(f"Document split into {len(chunks)} chunks")

            # Add document to vector database
            if chunks:
                self.vector_store.add_documents(chunks)

                # Save vector database
                self.vector_store.save(self.vector_db_path)

                logging.info(f"Document successfully added: {file_path}")
                return True
            else:
                logging.warning(f"No valid chunks after document processing: {file_path}")
                return False
        except Exception as e:
            logging.error(f"Error adding document: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def add_documents_from_directory(self, directory_path: Optional[str] = None) -> int:
        """
        Add all documents from directory to knowledge base

        Args:
            directory_path: Directory path (optional, defaults to knowledge base directory)

        Returns:
            Number of successfully added documents
        """
        directory_path = directory_path or self.knowledge_base_dir

        try:
            # Process all documents in directory
            chunks = self.document_processor.process_directory(directory_path)

            # Add documents to vector database
            self.vector_store.add_documents(chunks)

            # Save vector database
            self.vector_store.save(self.vector_db_path)

            logging.info(f"Documents from directory successfully added: {directory_path}")
            return len(chunks)
        except Exception as e:
            logging.error(f"Error adding documents from directory: {e}")
            return 0

    def query(self, query: str) -> str:
        """
        Query knowledge base to get relevant context

        Args:
            query: Query text

        Returns:
            Relevant context information
        """
        try:
            # Check if vector store is empty
            if not self.vector_store.documents:
                logging.warning("Knowledge base is empty, cannot query")
                return ""

            # Split user query into multiple context sentences
            # If query exceeds 20 characters, try to extract key sentences
            import re
            sentences = re.split(r'[.!?！？。]', query)  # Split by punctuation
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]  # Remove whitespace and short segments

            # If multiple sentences, use them as additional queries
            contexts = []

            # Query with original query first
            primary_context = self.retriever.get_context_for_query(query)
            if primary_context:
                contexts.append(primary_context)

            # If multiple sentences, also query each sentence
            if len(sentences) > 1:
                for sentence in sentences[:3]:  # Limit sentence count to avoid too many queries
                    if len(sentence) > 10:  # Only query longer sentences for context
                        sent_context = self.retriever.get_context_for_query(sentence)
                        if sent_context and sent_context not in contexts:
                            contexts.append(sent_context)

            # Combine all contexts
            if not contexts:
                return ""

            # If only one context, return directly
            if len(contexts) == 1:
                return contexts[0]

            # Merge multiple contexts
            combined_context = "Here are relevant information sections related to your question:\n\n"
            for i, ctx in enumerate(contexts, 1):
                # Remove title part from context to avoid repetition
                ctx_content = ctx.replace("Here is information relevant to your question:\n\n", "")
                combined_context += f"\n---- Related Material {i} ----\n{ctx_content}\n"

            return combined_context

        except Exception as e:
            logging.error(f"Error querying knowledge base: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return ""

    def get_prompt_with_context(self, query: str) -> str:
        """
        Get prompt with context

        Args:
            query: Query text

        Returns:
            Prompt with context
        """
        # Get context information for query
        context = self.query(query)

        # Combine local knowledge base and web search context
        combined_context = ""

        if context:
            combined_context += "Information found from local knowledge base:\n" + context + "\n\n"

        # If no context obtained, return query directly
        if not combined_context:
            return query

        # Format prompt with context
        prompt = f"{combined_context}Based on the above information, please answer the following question:\n\n{query}"

        return prompt
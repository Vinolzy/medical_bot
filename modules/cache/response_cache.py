"""
Response cache module
Responsible for storing and retrieving cached responses to improve system efficiency
"""
import os
import sys
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config

class ResponseCache:
    """
    Response cache class
    Uses vector similarity to find similar questions for intelligent caching
    """

    def __init__(self, cache_dir: Optional[str] = None,
                 embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 similarity_threshold: float = 0.85,
                 cache_ttl_days: int = 7):
        """
        Initialize response cache

        Args:
            cache_dir: Cache directory, default if not provided
            embedding_model: Model name for generating sentence embeddings
            similarity_threshold: Threshold for determining similar questions (0.0–1.0)
            cache_ttl_days: Cache time-to-live in days
        """
        # Set cache directory
        self.cache_dir = cache_dir or os.path.join(project_root, "data", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Set cache file paths
        self.cache_file = os.path.join(self.cache_dir, "response_cache.json")
        self.vector_file = os.path.join(self.cache_dir, "vectors.npy")

        # Set cache parameters
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = timedelta(days=cache_ttl_days)

        # Load cache
        self.cache_data = self._load_cache()

        # Initialize sentence model
        try:
            logging.info(f"Loading semantic model: {embedding_model}")
            self.model = SentenceTransformer(embedding_model)
            self.use_semantic = True
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}, falling back to basic cache")
            self.use_semantic = False

        # Load vector data
        self.question_vectors = {}
        if self.use_semantic:
            self._load_vectors()

    def _load_cache(self) -> Dict[str, Any]:
        """
        Load cached data

        Returns:
            Cache dictionary
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                logging.info(f"Loaded {len(cache_data)} cache records")

                # Clean expired cache
                return self._clean_expired_cache(cache_data)
            except Exception as e:
                logging.error(f"Failed to load cache: {e}")
                return {}
        else:
            logging.info("Cache file does not exist, creating new cache")
            return {}

    def _clean_expired_cache(self, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove expired cache records

        Args:
            cache_data: Cache dictionary

        Returns:
            Cleaned cache dictionary
        """
        current_time = datetime.now()
        cleaned_cache = {}

        for key, item in cache_data.items():
            # Check timestamp
            timestamp_str = item.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if current_time - timestamp <= self.cache_ttl:
                        cleaned_cache[key] = item
                except ValueError:
                    # Invalid timestamp, reset to current time
                    item['timestamp'] = current_time.isoformat()
                    cleaned_cache[key] = item
            else:
                # Missing timestamp, add current time
                item['timestamp'] = current_time.isoformat()
                cleaned_cache[key] = item

        if len(cache_data) != len(cleaned_cache):
            logging.info(f"Cleaned {len(cache_data) - len(cleaned_cache)} expired cache records")

        return cleaned_cache

    def _save_cache(self) -> None:
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(self.cache_data)} cache records")

            # Save vector data
            if self.use_semantic and self.question_vectors:
                self._save_vectors()
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")

    def _load_vectors(self) -> None:
        """Load question vector data"""
        if os.path.exists(self.vector_file):
            try:
                self.question_vectors = np.load(self.vector_file, allow_pickle=True).item()
                logging.info(f"Loaded {len(self.question_vectors)} question vectors")
            except Exception as e:
                logging.error(f"Failed to load vector data: {e}")
                self.question_vectors = {}

    def _save_vectors(self) -> None:
        """Save question vector data"""
        try:
            np.save(self.vector_file, self.question_vectors)
            logging.info(f"Saved {len(self.question_vectors)} question vectors")
        except Exception as e:
            logging.error(f"Failed to save vector data: {e}")

    def _get_cache_key(self, question: str) -> str:
        """
        Generate cache key from question

        Args:
            question: User question

        Returns:
            Cache key
        """
        return hashlib.md5(question.encode('utf-8')).hexdigest()

    def _get_question_embedding(self, question: str) -> np.ndarray:
        """
        Get question embedding vector

        Args:
            question: User question

        Returns:
            Embedding vector
        """
        if not self.use_semantic:
            return np.array([])

        # Check if vector is already computed
        question_hash = self._get_cache_key(question)
        if question_hash in self.question_vectors:
            return self.question_vectors[question_hash]

        # Compute question vector
        try:
            embedding = self.model.encode(question)
            self.question_vectors[question_hash] = embedding
            return embedding
        except Exception as e:
            logging.error(f"Error computing question embedding: {e}")
            return np.array([])

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0.0–1.0)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def find_similar_question(self, question: str) -> Tuple[str, float]:
        """
        Find the most similar cached question

        Args:
            question: User question

        Returns:
            (Most similar question, similarity score)
        """
        if not self.use_semantic or not self.cache_data:
            return ("", 0.0)

        # Get question vector
        question_vec = self._get_question_embedding(question)
        if len(question_vec) == 0:
            return ("", 0.0)

        best_match = ""
        best_score = 0.0

        for cached_item in self.cache_data.values():
            cached_question = cached_item.get('question', '')
            if not cached_question:
                continue

            # Get or compute cached question vector
            cached_question_hash = self._get_cache_key(cached_question)
            if cached_question_hash not in self.question_vectors:
                self.question_vectors[cached_question_hash] = self.model.encode(cached_question)

            cached_vec = self.question_vectors[cached_question_hash]
            similarity = self._calculate_similarity(question_vec, cached_vec)

            if similarity > best_score:
                best_score = similarity
                best_match = cached_question

        return (best_match, best_score)

    def get_response(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Try retrieving a cached response

        Args:
            question: User question

        Returns:
            Response dict or None if not found
        """
        # Check exact match first
        cache_key = self._get_cache_key(question)
        if cache_key in self.cache_data:
            cached_item = self.cache_data[cache_key]
            logging.info(f"Exact match found in cache: {question[:30]}...")
            cached_item['last_accessed'] = datetime.now().isoformat()
            self.cache_data[cache_key] = cached_item
            return cached_item

        # Find similar question
        if self.use_semantic:
            similar_question, similarity = self.find_similar_question(question)
            logging.info(f"Most similar question ({similarity:.2f}): {similar_question[:30]}...")

            if similarity >= self.similarity_threshold:
                similar_key = self._get_cache_key(similar_question)
                cached_item = self.cache_data[similar_key]
                logging.info(f"Using similar question cache (similarity: {similarity:.2f})")

                cached_item['last_accessed'] = datetime.now().isoformat()
                cached_item['similarity_matches'] = cached_item.get('similarity_matches', 0) + 1
                self.cache_data[similar_key] = cached_item

                result = dict(cached_item)
                result['similarity'] = similarity
                result['original_question'] = similar_question
                return result

        return None

    def add_response(self, question: str, response: str,
                     source_type: str = "direct", metadata: Dict[str, Any] = None) -> None:
        """
        Add response to cache

        Args:
            question: User question
            response: System response
            source_type: Response source type ("direct", "rag")
            metadata: Additional metadata
        """
        if not question or not response:
            return

        cache_key = self._get_cache_key(question)
        current_time = datetime.now()

        cache_item = {
            'question': question,
            'response': response,
            'source_type': source_type,
            'timestamp': current_time.isoformat(),
            'last_accessed': current_time.isoformat(),
            'access_count': 1
        }

        if metadata:
            cache_item['metadata'] = metadata

        self.cache_data[cache_key] = cache_item

        if self.use_semantic:
            self._get_question_embedding(question)

        self._save_cache()

        logging.info(f"Added new cache item: {question[:30]}...")

    def update_stats(self, question: str) -> None:
        """
        Update access statistics for a question

        Args:
            question: User question
        """
        cache_key = self._get_cache_key(question)
        if cache_key in self.cache_data:
            item = self.cache_data[cache_key]
            item['access_count'] = item.get('access_count', 0) + 1
            item['last_accessed'] = datetime.now().isoformat()
            self.cache_data[cache_key] = item

            if item['access_count'] % 10 == 0:
                self._save_cache()

    def clear_cache(self) -> None:
        """Clear all cache"""
        self.cache_data = {}
        self.question_vectors = {}
        self._save_cache()
        logging.info("All cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Stats dictionary
        """
        stats = {
            'total_entries': len(self.cache_data),
            'cache_size_bytes': os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0,
            'vector_size_bytes': os.path.getsize(self.vector_file) if os.path.exists(self.vector_file) else 0,
            'source_type_counts': {},
            'oldest_entry': None,
            'newest_entry': None,
            'most_accessed': None
        }

        if not self.cache_data:
            return stats

        # Count by source type
        for item in self.cache_data.values():
            source_type = item.get('source_type', 'unknown')
            stats['source_type_counts'][source_type] = stats['source_type_counts'].get(source_type, 0) + 1

        # Find oldest and newest
        sorted_by_time = sorted(
            [(k, v.get('timestamp', '')) for k, v in self.cache_data.items()],
            key=lambda x: x[1]
        )

        if sorted_by_time:
            oldest_key = sorted_by_time[0][0]
            newest_key = sorted_by_time[-1][0]
            stats['oldest_entry'] = {
                'question': self.cache_data[oldest_key].get('question', '')[:50],
                'timestamp': self.cache_data[oldest_key].get('timestamp', '')
            }
            stats['newest_entry'] = {
                'question': self.cache_data[newest_key].get('question', '')[:50],
                'timestamp': self.cache_data[newest_key].get('timestamp', '')
            }

        # Find most accessed
        most_accessed_key = max(
            self.cache_data.items(),
            key=lambda x: x[1].get('access_count', 0) if x[1].get('access_count') else 0,
            default=(None, {})
        )[0]

        if most_accessed_key:
            stats['most_accessed'] = {
                'question': self.cache_data[most_accessed_key].get('question', '')[:50],
                'access_count': self.cache_data[most_accessed_key].get('access_count', 0)
            }

        return stats

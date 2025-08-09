"""
Configuration File
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "local_llm")  # Default to local model
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Open Source LLM Settings
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL")
LOCAL_LLM_MODEL_PATH = os.getenv("LOCAL_LLM_MODEL_PATH")
LOCAL_LLM_TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE"))
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS"))
LOCAL_LLM_DEVICE = os.getenv("LOCAL_LLM_DEVICE", "cuda")  # 'cuda' or 'cpu'

# RAG Settings
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
EMBEDDING_CHUNK_SIZE = int(os.getenv("EMBEDDING_CHUNK_SIZE", "1000"))
EMBEDDING_CHUNK_OVERLAP = int(os.getenv("EMBEDDING_CHUNK_OVERLAP", "200"))

# Response Cache Settings
RESPONSE_CACHE_FILE = os.getenv("RESPONSE_CACHE_FILE", "data/cache/response_cache.json")
RESPONSE_CACHE_MAX_SIZE = int(os.getenv("RESPONSE_CACHE_MAX_SIZE", "100"))
RESPONSE_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("RESPONSE_CACHE_SIMILARITY_THRESHOLD", "0.85"))

# Text-to-Speech Settings
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
DEFAULT_VOICE_ID = os.getenv("DEFAULT_VOICE_ID", "")  # ElevenLabs voice ID

# Vision Model Settings
VISION_CONFIG = {
    "enabled": os.getenv("VISION_ENABLED", "True").lower() in ('true', '1', 't'),
    "model": os.getenv("VISION_MODEL", "gpt-4o"),  # Vision uses OpenAI API
    "max_tokens": int(os.getenv("VISION_MAX_TOKENS", "1000")),
    "supported_formats": ['.jpg', '.jpeg', '.png', '.webp', '.gif']
}

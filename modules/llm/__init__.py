from .openai_client import OpenAIClient

import os
import logging
import sys
import config

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)


logger = logging.getLogger(__name__)


def create_llm_client():
    from .llama_index_adapter import LlamaIndexAdapter
    return LlamaIndexAdapter(
        model_name_or_path=config.LOCAL_LLM_MODEL_PATH,
        device=config.LOCAL_LLM_DEVICE,
        temperature=config.LOCAL_LLM_TEMPERATURE,
        max_tokens=config.LOCAL_LLM_MAX_TOKENS,
    )


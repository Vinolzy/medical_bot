"""
Local LLM Model Interface Module
Supports GPU-accelerated Llama 3.2 model with DPO
"""
import os
import logging
import torch
from typing import List, Dict, Any, Optional
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config


class LocalLLM:
    """
    Local LLM Model Class
    For interacting with Llama 3.2 and other local models
    """

    def __init__(
            self,
            model_name_or_path: str = None,
            device: str = None,
            temperature: float = None,
            max_tokens: int = None
    ):
        """
        Initialize local LLM model

        Args:
            model_name_or_path: Model name or path, defaults to config setting
            device: Execution device ('cuda' or 'cpu'), defaults to config setting
            temperature: Generation temperature, defaults to config setting
            max_tokens: Maximum generation tokens, defaults to config setting
        """
        self.model_name_or_path = model_name_or_path or config.LOCAL_LLM_MODEL_PATH
        self.device = device or config.LOCAL_LLM_DEVICE
        self.temperature = temperature or config.LOCAL_LLM_TEMPERATURE
        self.max_tokens = max_tokens or config.LOCAL_LLM_MAX_TOKENS

        # Check if CUDA is available
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, switching to CPU")
            self.device = "cpu"

        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer using the specified configuration"""
        try:
            logger.info(f"Loading Llama model: {self.model_name_or_path}")

            # Load tokenizer - using meta-llama/Llama-3.2-3B-Instruct as base
            # You may need to adjust this based on your actual tokenizer path
            tokenizer_name = "meta-llama/Llama-3.2-3B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with specified configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                local_files_only=True,
                low_cpu_mem_usage=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

            # Create pipeline for text generation
            self.pipeline = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
            )

            logger.info(f"Llama model successfully loaded with device_map='auto'")

        except Exception as e:
            logger.error(f"Error loading Llama model: {e}")
            raise

    def generate_response(self, messages, system_prompt=None, temperature=None, max_tokens=None) -> str:
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")

        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        # 1) Assemble HuggingFace messages
        formatted = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        formatted.extend(messages)

        # 2) Render into a string using the chat template
        rendered = self.tokenizer.apply_chat_template(
            formatted, tokenize=False, add_generation_prompt=True
        )

        # 3) Generate (only new content; sampling is off if temperature==0)
        outputs = self.pipeline(
            rendered,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=bool(temperature and temperature > 0),
            temperature=temperature,
            return_full_text=False,
        )

        return outputs[0]["generated_text"]

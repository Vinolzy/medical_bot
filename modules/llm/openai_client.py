"""
OpenAI Client Module
Responsible for communication with OpenAI API and response generation
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import configuration
import sys
import os

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config

class OpenAIClient:
    """
    OpenAI Client Class, responsible for handling communication with OpenAI API
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client

        Args:
            api_key: OpenAI API key, retrieves from environment variables if not provided
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.error("OpenAI API key not set, please set OPENAI_API_KEY in .env file")
            raise ValueError("OpenAI API key is required")

        # Set default model and parameters
        self.default_model = config.OPENAI_MODEL
        self.temperature = float(config.OPENAI_TEMPERATURE)
        self.max_tokens = 2000  # Default max output tokens

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        logging.info(f"OpenAI client initialized with model: {self.default_model}")

        # Additional check: only OpenAI models should use OpenAI API
        if self.default_model == "local_llm":
            logging.warning("Warning: OpenAI client initialized with local_llm model, but OpenAI client cannot use local models")

    def generate_response(self,
                         messages: List[Dict[str, Any]],
                         system_prompt: Optional[str] = None,
                         model: Optional[str] = None,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """
        Generate AI response

        Args:
            messages: Message history list
            system_prompt: System prompt (optional)
            model: Model name (optional, defaults to configured model)
            temperature: Temperature parameter (optional, defaults to configured value)
            max_tokens: Maximum tokens (optional, defaults to configured value)

        Returns:
            Generated response text
        """
        if config.OPENAI_MODEL == "local_llm":
            logging.error("Attempting to use OpenAI client for response generation, but config specifies local_llm. This may indicate the system is not properly using LocalLLMClient.")

        try:
            # Prepare request messages
            request_messages = []

            # Add system prompt (if provided)
            if system_prompt:
                request_messages.append({"role": "system", "content": system_prompt})

            # Add message history
            request_messages.extend(messages)

            # Determine model to use
            use_model = model or self.default_model
            if use_model == "local_llm":
                # If local_llm is specified, use fallback OpenAI model
                logging.warning("Model set to local_llm but using OpenAI client, switching to gpt-4o-mini")
                use_model = "gpt-4o-mini"

            # Log request info
            logging.info(f"OpenAI API request with model: {use_model}")

            # Call API
            response = self.client.chat.completions.create(
                model=use_model,
                messages=request_messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )

            # Extract response text
            response_text = response.choices[0].message.content.strip()

            return response_text

        except Exception as e:
            logging.error(f"OpenAI API request failed: {e}")
            # In production, might want to return a graceful error message
            # But during development, letting errors propagate might be more useful
            raise

    def generate_vision_response(self,
                                messages: List[Dict[str, Any]],
                                system_prompt: Optional[str] = None,
                                model: Optional[str] = None) -> str:
        """
        Generate response with vision understanding

        Args:
            messages: Message list containing images and text
            system_prompt: System prompt
            model: Model to use, defaults to vision model

        Returns:
            Generated response text
        """
        try:
            # Use specified model or default vision model
            vision_model = model or config.VISION_CONFIG.get("model", self.default_model)

            # Prepare message list
            request_messages = []

            # Add system prompt (if provided)
            if system_prompt:
                request_messages.append({"role": "system", "content": system_prompt})

            # Add message history
            request_messages.extend(messages)

            # Log request info
            logging.info(f"OpenAI Vision API request with model: {vision_model}")

            # Call API
            response = self.client.chat.completions.create(
                model=vision_model,
                messages=request_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract response text
            response_text = response.choices[0].message.content.strip()

            return response_text

        except Exception as e:
            logging.error(f"OpenAI Vision API request failed: {e}")
            return f"Sorry, I encountered an issue processing the image: {str(e)}"
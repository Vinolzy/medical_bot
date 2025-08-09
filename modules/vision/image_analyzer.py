"""
Image Analysis Module
Uses GPT-4o-mini for image understanding and analysis
"""
import os
import sys
import base64
import logging
from typing import Dict, Any, Optional, List
from PIL import Image
import io

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config
from modules.llm.openai_client import OpenAIClient

class ImageAnalyzer:
    """
    Image Analysis Class
    Uses OpenAI's GPT-4o-mini for visual understanding
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize image analyzer

        Args:
            api_key: OpenAI API key, retrieves from config if not provided
        """
        self.openai_client = OpenAIClient(api_key=api_key)
        self.model = config.VISION_CONFIG.get("model", "gpt-4o")  # Use vision-capable model
        logging.info("Image analyzer initialized")

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string

        Args:
            image_path: Image file path

        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Image encoding error: {e}")
            raise

    def resize_image_if_needed(self, image_path: str, max_size: int = 4 * 1024 * 1024) -> str:
        """
        Resize image if it's too large

        Args:
            image_path: Image file path
            max_size: Maximum file size in bytes

        Returns:
            Path to possibly resized image
        """
        try:
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size <= max_size:
                return image_path

            logging.info(f"Image size ({file_size} bytes) exceeds limit, resizing")

            # Open and resize image
            img = Image.open(image_path)

            # Calculate resize ratio
            ratio = (max_size / file_size) ** 0.5
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)

            # Resize
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # Save resized image
            filename, ext = os.path.splitext(image_path)
            resized_path = f"{filename}_resized{ext}"
            resized_img.save(resized_path, quality=85)

            logging.info(f"Image resized and saved to: {resized_path}")
            return resized_path
        except Exception as e:
            logging.error(f"Error resizing image: {e}")
            return image_path  # Return original path

    def analyze_image(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze image content

        Args:
            image_path: Image file path
            prompt: Custom prompt text (optional)

        Returns:
            Dictionary containing analysis results
        """
        try:
            logging.info(f"Starting image analysis: {image_path}")

            # Use default prompt if none provided
            if not prompt:
                prompt = "Describe this image in detail. Focus on objects, people, scenes, or text visible in the image."

            # Check if file exists
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            # Resize image if necessary
            resized_path = self.resize_image_if_needed(image_path)

            # Encode image to base64
            base64_image = self.encode_image_to_base64(resized_path)

            # Remove temporary file if resized
            if resized_path != image_path:
                os.remove(resized_path)

            # Build request messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            # Call OpenAI API to analyze image
            try:
                system_prompt = """You are an AI assistant with visual capabilities.
Analyze the provided image carefully and provide clear, detailed descriptions.
Describe what you see directly without prefacing with phrases like "This image shows".
If the image contains sensitive or inappropriate content, politely decline detailed description and provide only general observations."""

                response = self.openai_client.generate_vision_response(messages, system_prompt, model=self.model)
            except Exception as e:
                logging.error(f"OpenAI vision API error: {e}")
                import traceback
                logging.error(traceback.format_exc())
                return {
                    "success": False,
                    "error": f"Vision model response error: {str(e)}"
                }

            # Log analysis completion
            logging.info(f"Image analysis completed: {image_path}")

            return {
                "success": True,
                "analysis": response,
                "prompt": prompt
            }
        except Exception as e:
            logging.error(f"Image analysis error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
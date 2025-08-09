"""
Speech Service Module
Provides speech recognition and text-to-speech functionality
"""
import os
import sys
import io
import json
import logging
from pathlib import Path
import tempfile
import wave
import numpy as np
import requests
from typing import Tuple, Optional, Union, Dict, Any

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config
from openai import OpenAI

class SpeechService:
    """
    Speech Service Class
    Provides speech-to-text (STT) and text-to-speech (TTS) functionality
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize speech service

        Args:
            api_key: OpenAI API key, retrieves from config if not provided
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.api_key)
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")

        # Check configuration
        if not self.api_key:
            logging.warning("OpenAI API key not set, speech recognition will be unavailable")

        # Ensure temp directory exists
        self.temp_dir = Path(project_root) / "data" / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def speech_to_text(self, audio_data: Union[bytes, str],
                      mime_type: str = "audio/wav") -> Dict[str, Any]:
        """
        Convert speech to text

        Args:
            audio_data: Audio data (bytes) or audio file path
            mime_type: Audio MIME type

        Returns:
            Dictionary containing transcribed text, format: {'text': 'transcribed_text', 'status': 'success/failed'}
        """
        if not self.api_key:
            return {"text": "", "status": "failed", "error": "OpenAI API key not set"}

        try:
            print("audio:", audio_data)
            # If input is file path, read the file
            if isinstance(audio_data, str):
                with open(audio_data, "rb") as audio_file:
                    audio_bytes = audio_file.read()
            else:
                audio_bytes = audio_data

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=self.temp_dir) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            # Transcribe using OpenAI's Whisper model
            with open(temp_file_path, "rb") as audio_file:
                response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    # language="en"
                )

            # Clean up temporary file
            os.unlink(temp_file_path)

            return {"text": response.text, "status": "success"}

        except Exception as e:
            logging.error(f"Speech-to-text failed: {str(e)}")
            return {"text": "", "status": "failed", "error": str(e)}

    def text_to_speech(self, text: str, voice_id: Optional[str] = None,
                       use_elevenlabs: bool = False) -> Tuple[bytes, str]:
        """
        Convert text to speech

        Args:
            text: Text to convert
            voice_id: Voice ID (if using ElevenLabs)
            use_elevenlabs: Whether to use ElevenLabs for speech synthesis

        Returns:
            Audio data (bytes) and MIME type
        """
        # For simplicity, will only use OpenAI TTS
        return self._openai_tts(text)

    def _openai_tts(self, text: str, voice: str = "alloy") -> Tuple[bytes, str]:
        """
        Generate speech using OpenAI's TTS API

        Args:
            text: Text to convert
            voice: Voice type (alloy, echo, fable, onyx, nova, shimmer)

        Returns:
            Audio data (bytes) and MIME type
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not set")

        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )

            # Get audio data
            audio_data = response.content

            return audio_data, "audio/mp3"

        except Exception as e:
            logging.error(f"OpenAI text-to-speech failed: {str(e)}")
            raise



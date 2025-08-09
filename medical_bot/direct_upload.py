"""
File upload handling module
Provides file upload and basic processing functions
"""
import os
import logging
from typing import Tuple, Optional
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Set logging
logging.basicConfig(level=logging.INFO)

def handle_uploaded_file(file, is_image: bool = False) -> Tuple[bool, str]:
    """
    Handle uploaded file

    Args:
        file: Uploaded file object
        is_image: Whether the file is an image

    Returns:
        (Success flag, file path or error message)
    """
    try:
        # Get file name and extension
        file_name = file.name
        file_ext = os.path.splitext(file_name)[1].lower()

        # Check file type
        if is_image:
            supported_formats = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
            if file_ext not in supported_formats:
                return False, f"Unsupported image format. Please upload an image in {', '.join(supported_formats)} format."

            # Set image save path
            target_dir = os.path.join(BASE_DIR, "data", "images")
        else:
            supported_formats = ['.pdf', '.txt', '.docx', '.doc']
            if file_ext not in supported_formats:
                return False, f"Unsupported file format. Please upload a file in {', '.join(supported_formats)} format."

            # Set document save path
            target_dir = os.path.join(BASE_DIR, "data", "knowledge_base")

        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Generate a safe file name
        import re
        safe_filename = re.sub(r'[\\/*?:"<>|]', '_', file_name)
        file_path = os.path.join(target_dir, safe_filename)

        # Save file
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        logging.info(f"File saved to: {file_path}")
        return True, file_path

    except Exception as e:
        logging.error(f"Error while processing uploaded file: {e}")
        return False, str(e)

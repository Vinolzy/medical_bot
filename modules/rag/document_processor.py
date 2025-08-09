"""
Document Processor Module
Handles document loading and chunking
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Libraries for processing different document types
import docx
import PyPDF2
import re

class DocumentProcessor:
    """
    Document Processor Class, handles document loading and chunking
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor

        Args:
            chunk_size: Document chunk size
            chunk_overlap: Overlapping characters between adjacent chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logging.info(f"Document processor initialized, chunk size: {chunk_size}, overlap size: {chunk_overlap}")

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process single file and return document chunks

        Args:
            file_path: File path

        Returns:
            List of document chunks, each chunk is a dictionary containing text and metadata
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                return []

            # Get file type
            file_ext = os.path.splitext(file_path)[1].lower()

            # Load file content
            if file_ext == '.txt':
                text = self._load_text_file(file_path)
            elif file_ext == '.pdf':
                text = self._load_pdf_file(file_path)
            elif file_ext in ['.docx', '.doc']:
                text = self._load_docx_file(file_path)
            else:
                logging.error(f"Unsupported file type: {file_ext}")
                return []

            # Return empty list if file content is empty
            if not text:
                logging.warning(f"File content is empty: {file_path}")
                return []

            # Split text into chunks
            chunks = self._split_text(text)

            # Create document chunks
            doc_chunks = []
            file_name = os.path.basename(file_path)

            for i, chunk in enumerate(chunks):
                # Remove blank lines and excess whitespace
                chunk = self._clean_text(chunk)

                # Add to result if chunk is not empty
                if chunk:
                    doc_chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source": file_name,
                            "chunk_id": i,
                            "file_path": file_path
                        }
                    })

            logging.info(f"File {file_path} processed into {len(doc_chunks)} chunks")
            return doc_chunks

        except Exception as e:
            logging.error(f"Error processing file: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []

    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all files in directory and return all document chunks

        Args:
            directory_path: Directory path

        Returns:
            List of all document chunks
        """
        all_chunks = []

        try:
            # Check if directory exists
            if not os.path.isdir(directory_path):
                logging.error(f"Directory not found: {directory_path}")
                return []

            # Get all files in directory
            files = []
            for root, _, filenames in os.walk(directory_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in ['.txt', '.pdf', '.docx', '.doc']:
                        files.append(file_path)

            # Process each file
            for file_path in files:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)

            logging.info(f"Directory {directory_path} with {len(files)} files processed, total {len(all_chunks)} chunks")
            return all_chunks

        except Exception as e:
            logging.error(f"Error processing directory: {e}")
            return []

    def _load_text_file(self, file_path: str) -> str:
        """
        Load text file

        Args:
            file_path: File path

        Returns:
            File content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try other encodings
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    text = f.read()
                return text
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='big5') as f:
                        text = f.read()
                    return text
                except:
                    logging.error(f"Cannot decode text file: {file_path}")
                    return ""
        except Exception as e:
            logging.error(f"Error loading text file: {e}")
            return ""

    def _load_pdf_file(self, file_path: str) -> str:
        """
        Load PDF file

        Args:
            file_path: File path

        Returns:
            File content
        """
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text
        except Exception as e:
            logging.error(f"Error loading PDF file: {e}")
            return ""

    def _load_docx_file(self, file_path: str) -> str:
        """
        Load DOCX file

        Args:
            file_path: File path

        Returns:
            File content
        """
        try:
            doc = docx.Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            logging.error(f"Error loading DOCX file: {e}")
            return ""

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Return directly if text length is smaller than chunk size
        if len(text) <= self.chunk_size:
            return [text]

        # Handle excess newlines and spaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{3,}', ' ', text)

        # Split by paragraphs first
        paragraphs = text.split('\n\n')

        # Initialize result
        chunks = []
        current_chunk = ""

        # Iterate through paragraphs
        for para in paragraphs:
            # If adding paragraph to current chunk doesn't exceed max length, add to current chunk
            if len(current_chunk) + len(para) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # If paragraph itself exceeds chunk size, need further splitting
                if len(para) > self.chunk_size:
                    # Save current chunk first
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""

                    # Split long paragraph by sentences
                    sentences = re.split(r'(?<=[。！？.!?])', para)

                    # Combine sentences into chunks
                    temp_chunk = ""
                    for sentence in sentences:
                        if not sentence:
                            continue

                        if len(temp_chunk) + len(sentence) <= self.chunk_size:
                            temp_chunk += sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)

                            # If sentence itself exceeds chunk size, split by characters
                            if len(sentence) > self.chunk_size:
                                sentence_chunks = [sentence[i:i+self.chunk_size] for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap)]
                                chunks.extend(sentence_chunks)
                                temp_chunk = ""
                            else:
                                temp_chunk = sentence

                    # Save last temporary chunk
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    # Current chunk has reached max length, save and start new chunk
                    chunks.append(current_chunk)
                    current_chunk = para

        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Handle chunk overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i < len(chunks) - 1 and len(chunks[i]) + self.chunk_overlap <= self.chunk_size:
                    # Add beginning of next chunk to end of current chunk
                    next_chunk_start = chunks[i + 1][:min(self.chunk_overlap, len(chunks[i + 1]))]
                    overlapped_chunks.append(chunks[i] + "\n" + next_chunk_start)
                else:
                    overlapped_chunks.append(chunks[i])

            return overlapped_chunks

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing excess whitespace and blank lines

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Handle excess newlines and spaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{3,}', ' ', text)
        return text.strip()
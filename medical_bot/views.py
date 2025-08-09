"""
Django views module
Provides API endpoints
"""
import os
import sys
import json
import logging
from datetime import datetime
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import tempfile
import shutil

# Import file upload handling module
from medical_bot.direct_upload import handle_uploaded_file

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import config
from modules.llm import create_llm_client
from modules.rag.rag_manager import RAGManager
from modules.cache.response_cache import ResponseCache
from modules.speech.speech_service import SpeechService
from modules.vision.image_analyzer import ImageAnalyzer

# Set detailed log format early
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize LLM client (use factory function to decide between local model or OpenAI API)
logging.info("Creating LLM client...")
llm_client = create_llm_client()
logging.info(f"LLM client created, type: {type(llm_client).__name__}")

# Initialize RAG manager
rag_manager = RAGManager()

# Initialize response cache
response_cache = ResponseCache()

# Initialize speech service
speech_service = SpeechService()

# Initialize image analyzer
image_analyzer = ImageAnalyzer()

# System prompt
SYSTEM_PROMPT = "Respond with deep compassion and empathy, offering comfort and understanding. You may add fitting emojis to enhance the emotional tone."

@csrf_exempt
def index(request):
    """Home view"""
    return JsonResponse({"status": "success", "message": "Medical AI Assistant API is running"})

@csrf_exempt
def health_check(request):
    """Health check endpoint"""
    return JsonResponse({"status": "healthy"})

@csrf_exempt
def medicalbot_api(request):
    """medicalbot API endpoint."""
    if request.method != 'POST':
        return JsonResponse({"status": "error", "message": "Only POST requests supported"}, status=405)

    try:
        # Parse request data
        data = json.loads(request.body)
        message = data.get('message', '')
        conversation_id = data.get('conversation_id', '')
        use_rag = data.get('use_rag', True)
        use_cache = data.get('use_cache', True)

        # Check cache first if enabled
        if use_cache:
            cached_response = response_cache.get_response(message)
            if cached_response:
                response_text = cached_response['response']
                metadata = {"source": "cache"}

                # Similarity metadata passthrough
                if 'similarity' in cached_response:
                    metadata.update({
                        "cache_type": "similar",
                        "similarity": cached_response.get('similarity', 0.0),
                        "original_question": cached_response.get('original_question', '')
                    })
                else:
                    metadata["cache_type"] = "exact"

                response_cache.update_stats(message)

                return JsonResponse({
                    "status": "success",
                    "message": response_text,
                    "conversation_id": conversation_id,
                    "from_cache": True,
                    "metadata": metadata
                })

        # Default: direct source unless RAG is used
        enhanced_message = message
        source_info = {"type": "direct"}

        # RAG enrichment
        if use_rag:
            enhanced_message = rag_manager.get_prompt_with_context(message)

            if enhanced_message != message:
                logging.info(f"Using enhanced prompt: \n{enhanced_message[:300]}...")

                # Build system prompt for RAG mode
                system_prompt = SYSTEM_PROMPT + "\nUse the provided materials to answer questions."

                # Send ONLY the current turn to the LLM
                response_text = llm_client.generate_response(
                    [{"role": "user", "content": enhanced_message}],
                    system_prompt
                )
                source_info = {"type": "rag"}

            else:
                # No RAG augmentation found; fall back to normal system prompt
                response_text = llm_client.generate_response(
                    [{"role": "user", "content": message}],
                    SYSTEM_PROMPT
                )
        else:
            # RAG disabled
            response_text = llm_client.generate_response(
                [{"role": "user", "content": message}],
                SYSTEM_PROMPT
            )

        # Cache response if enabled
        if use_cache:
            source_type_map = {"rag": "rag", "direct": "direct"}
            source_type = source_type_map.get(source_info["type"], "direct")

            response_cache.add_response(
                question=message,
                response=response_text,
                source_type=source_type,
                metadata={
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "use_rag": use_rag,
                }
            )

        return JsonResponse({
            "status": "success",
            "message": response_text,
            "conversation_id": conversation_id,
            "source_info": source_info
        })

    except Exception as e:
        logging.error(f"Request processing error: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=400)


@csrf_exempt
def handle_image_upload(file):
    """Handle image upload and analyze"""
    try:
        # Handle image upload via direct_upload module
        success, result = handle_uploaded_file(file, is_image=True)

        if not success:
            return JsonResponse({
                "status": "error",
                "message": result
            }, status=400)

        image_path = result

        # Analyze image using the image analyzer
        logging.info(f"Starting image analysis: {image_path}")
        analysis_result = image_analyzer.analyze_image(image_path)

        if not analysis_result["success"]:
            return JsonResponse({
                "status": "error",
                "message": f"Error analyzing image: {analysis_result.get('error', '')}"
            }, status=400)

        # Return analysis result
        return JsonResponse({
            "status": "success",
            "message": f"Image {file.name} analyzed successfully",
            "analysis": analysis_result["analysis"],
            "file_path": image_path
        })
    except Exception as e:
        logging.error(f"Error handling image upload: {e}")
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)

@csrf_exempt
def analyze_image(request):
    """Image analysis API endpoint"""
    if request.method == 'POST':
        try:
            # Check if an image file is provided
            if 'file' not in request.FILES and 'image' not in request.FILES:
                return JsonResponse({"status": "error", "message": "Please provide an image file"}, status=400)

            # Get file, supporting 'file' or 'image' param name
            image_file = request.FILES.get('file') or request.FILES.get('image')

            # Handle image upload
            return handle_image_upload(image_file)

        except Exception as e:
            logging.error(f"Image analysis error: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "Only POST requests supported"}, status=405)

@csrf_exempt
def upload_file(request):
    """File upload endpoint"""
    if request.method == 'POST':
        try:
            file = request.FILES.get('file')
            if not file:
                return JsonResponse({"status": "error", "message": "No file provided"}, status=400)

            # Get file name and extension
            file_name = file.name
            file_ext = os.path.splitext(file_name)[1].lower()

            # Check if the file is an image
            is_image = file_ext in config.VISION_CONFIG.get("supported_formats", ['.jpg', '.jpeg', '.png', '.webp', '.gif'])

            # If it's an image, call the image analysis function
            if is_image:
                return handle_image_upload(file)

            # Check document type
            if file_ext not in ['.pdf', '.txt', '.docx', '.doc']:
                return JsonResponse({
                    "status": "error",
                    "message": "Unsupported file type. Please upload PDF, TXT, Word file, or a supported image format"
                }, status=400)

            # Handle file upload
            success, result = handle_uploaded_file(file, is_image=False)

            if not success:
                return JsonResponse({
                    "status": "error",
                    "message": result
                }, status=400)

            file_path = result

            # Try adding the file to the knowledge base
            try:
                success = rag_manager.add_document(file_path)
                if success:
                    return JsonResponse({
                        "status": "success",
                        "message": f"File {file_name} uploaded and added to knowledge base successfully",
                        "file_path": file_path
                    })
                else:
                    return JsonResponse({
                        "status": "partial_success",
                        "message": f"File {file_name} uploaded, but failed to add to knowledge base"
                    })
            except Exception as e:
                logging.error(f"Error adding file to knowledge base: {e}")
                return JsonResponse({
                    "status": "partial_success",
                    "message": f"File {file_name} uploaded, but failed to add to knowledge base: {str(e)}"
                })

        except Exception as e:
            logging.error(f"Error handling file upload: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "Only POST requests supported"}, status=405)

@csrf_exempt
def rebuild_knowledge_base(request):
    """Rebuild knowledge base endpoint"""
    if request.method == 'POST':
        try:
            # Create new RAG manager
            global rag_manager
            rag_manager = RAGManager()

            # Add all documents from the knowledge base directory
            count = rag_manager.add_documents_from_directory()

            return JsonResponse({
                "status": "success",
                "message": f"Knowledge base rebuilt. Added {count} document chunks"
            })
        except Exception as e:
            logging.error(f"Error rebuilding knowledge base: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "Only POST requests supported"}, status=405)

@csrf_exempt
def cache_stats(request):
    """Cache statistics endpoint"""
    if request.method == 'GET':
        try:
            # Get cache statistics
            stats = response_cache.get_cache_stats()

            return JsonResponse({
                "status": "success",
                "stats": stats
            })
        except Exception as e:
            logging.error(f"Error getting cache stats: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "Only GET requests supported"}, status=405)

@csrf_exempt
def clear_cache(request):
    """Clear cache endpoint"""
    if request.method == 'POST':
        try:
            # Clear cache
            response_cache.clear_cache()

            return JsonResponse({
                "status": "success",
                "message": "Cache cleared successfully"
            })
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "Only POST requests supported"}, status=405)


@csrf_exempt
def speech_to_text(request):
    """Speech-to-text endpoint"""
    if request.method == 'POST':
        try:
            # Check if audio file is provided
            if 'audio' not in request.FILES:
                return JsonResponse({
                    "status": "error",
                    "message": "No audio file provided"
                }, status=400)

            audio_file = request.FILES['audio']
            audio_data = audio_file.read()

            # Check if audio is empty
            if not audio_data:
                return JsonResponse({
                    "status": "error",
                    "message": "Audio file is empty"
                }, status=400)

            # Convert audio using speech service
            result = speech_service.speech_to_text(audio_data)

            if result["status"] == "success":
                return JsonResponse({
                    "status": "success",
                    "text": result["text"]
                })
            else:
                return JsonResponse({
                    "status": "error",
                    "message": result.get("error", "Speech-to-text failed")
                }, status=500)

        except Exception as e:
            logging.error(f"Error processing speech-to-text request: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "Only POST requests supported"}, status=405)

@csrf_exempt
def text_to_speech(request):
    """Text-to-speech endpoint"""
    if request.method == 'POST':
        try:
            # Parse request data
            data = json.loads(request.body)
            text = data.get('text', '')

            if not text:
                return JsonResponse({
                    "status": "error",
                    "message": "No text provided"
                }, status=400)

            # Use ElevenLabs or default OpenAI
            use_elevenlabs = data.get('use_elevenlabs', False)
            voice_id = data.get('voice_id', None)

            # Generate audio using speech service
            try:
                audio_data, mime_type = speech_service.text_to_speech(
                    text,
                    voice_id=voice_id,
                    use_elevenlabs=use_elevenlabs
                )

                # Create HTTP response
                response = HttpResponse(audio_data, content_type=mime_type)
                response['Content-Disposition'] = 'attachment; filename="speech.mp3"'
                return response

            except Exception as e:
                logging.error(f"Error generating speech: {e}")
                return JsonResponse({
                    "status": "error",
                    "message": f"Error generating speech: {str(e)}"
                }, status=500)

        except Exception as e:
            logging.error(f"Error processing text-to-speech request: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "Only POST requests supported"}, status=405)

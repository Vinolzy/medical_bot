import os
import sys
import requests
import shutil
import traceback
import chainlit as cl

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Django API POST URL
DJANGO_API_BASE_URL = os.environ.get("DJANGO_API_BASE_URL")

@cl.on_chat_start
async def on_chat_start():
    """Trigger when a new chat is created, a new user_session is created for different dialogues."""
    import uuid
    conversation_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", conversation_id) # store in Django server
    cl.user_session.set("message_history", [])
    cl.user_session.set("use_rag", True)
    cl.user_session.set("use_cache", False)

    welcome_elements = [
        cl.Text(name="voice_status", content="Tip: Use the mic button for voice input"),
        cl.Text(name="rag_status", content="Tip: Use `/rag on` or `/rag off` to enable/disable knowledge base reference"),
        cl.Text(name="cache_status", content="Tip: Use `/cache on` or `/cache off` to enable/disable answer caching"),
    ]

    # Prepare welcome message
    welcome_msg = (
        "Hello! I am a medical AI assistant.\n\n"
        "● You can directly type your question\n"
        "● Use the mic button for voice input\n"
        "● Use the paperclip button (bottom left) to upload files\n"
        "● Uploaded files are automatically added to the knowledge base\n"
        "● Answer caching is enabled by default for faster responses to similar questions\n"
    )

    # Send welcome message
    await cl.Message(
        content=welcome_msg,
        elements=welcome_elements
    ).send()



async def get_cache_stats():
    """Get and display cache statistics"""
    try:
        response = requests.get(f"{DJANGO_API_BASE_URL}/cache_stats/")
        if response.status_code == 200:
            stats = response.json().get("stats", {})

            text = f"Cache Statistics:\n"
            text += f"• Total entries: {stats.get('total_entries', 0)}\n"
            text += f"• Cache size: {stats.get('cache_size_bytes', 0) // 1024} KB\n"

            # Source distribution
            if 'source_type_counts' in stats:
                text += "\nSource Distribution:\n"
                source_names = {"direct": "Direct", "rag": "RAG"}
                for source, count in stats.get('source_type_counts', {}).items():
                    name = source_names.get(source, source)
                    text += f"• {name}: {count}\n"

            # Most accessed question
            if stats.get('most_accessed'):
                ma = stats['most_accessed']
                text += f"\nMost accessed: {ma.get('question', '')}\n"
                text += f"• Access count: {ma.get('access_count', 0)}"

            await cl.Message(content=text).send()
        else:
            await cl.Message(content=f"Failed to get cache stats: {response.status_code}").send()
    except Exception as e:
        await cl.Message(content=f"Cache stats error: {str(e)}").send()


async def clear_cache():
    """Clear the cache"""
    try:
        response = requests.post(f"{DJANGO_API_BASE_URL}/clear_cache/")
        if response.status_code == 200:
            await cl.Message(content="Cache cleared successfully.").send()
        else:
            await cl.Message(content=f"Failed to clear cache: {response.status_code}").send()
    except Exception as e:
        await cl.Message(content=f"Clear cache error: {str(e)}").send()


async def show_voice_help():
    """Show voice feature help"""
    help_text = """Voice Feature Guide:

    1. Click the microphone icon next to the input field
    2. Speak into your microphone
    3. Click stop or wait for auto-stop
    4. Speech will be automatically converted to text"""

    await cl.Message(content=help_text).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle chat messages from Chainlit UI.
    """
    # get config
    conversation_id = cl.user_session.get("conversation_id")
    use_rag = cl.user_session.get("use_rag")
    use_cache = cl.user_session.get("use_cache")

    # check if a file is uploaded
    if message.elements:
        print(f"file details: {len(message.elements)}")
        for elem in message.elements:
            print(f"file type: {type(elem)}")
            await process_file(elem)
        return

    # Command mapping for toggle features
    toggle_commands = {
        "/rag on": ("use_rag", True, "RAG enabled - I'll reference uploaded files."),
        "/rag off": ("use_rag", False, "RAG disabled - I won't reference uploaded files."),
        "/cache on": ("use_cache", True, "Cache enabled - Quick responses for similar questions."),
        "/cache off": ("use_cache", False, "Cache disabled - Processing each question fresh."),
    }

    cmd = message.content.lower()

    # Handle toggle commands
    if cmd in toggle_commands:
        key, value, msg = toggle_commands[cmd]
        cl.user_session.set(key, value)
        await cl.Message(content=msg).send()
        return

    # Handle API-based commands
    api_commands = {
        "/cache stats": get_cache_stats,
        "/cache clear": clear_cache,
        "/voice": show_voice_help
    }

    if cmd in api_commands:
        await api_commands[cmd]()
        return

    thinking_msg = cl.Message(content="Pondering...")
    await thinking_msg.send()

    # Prepare payload. LlamaIndex memory owns the history.
    request_data = {
        "message": message.content,
        "conversation_id": conversation_id,
        "use_rag": use_rag,
        "use_cache": use_cache,
    }

    try:
        response = requests.post(
            f"{DJANGO_API_BASE_URL}/medicalbot/",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        await thinking_msg.remove()

        if response.status_code == 200:
            response_data = response.json()
            response_text = response_data.get("message", "")

            # Generate status elements
            elements = []

            # Cache indicators
            if response_data.get("from_cache"):
                metadata = response_data.get("metadata", {})
                if metadata.get("cache_type") == "similar":
                    similarity = metadata.get("similarity", 0.0)
                    elements.append(cl.Text(
                        name="similar_cache",
                        content=f"✓ From similar question cache (similarity: {similarity:.2f})"
                    ))
                else:
                    elements.append(cl.Text(name="cache", content="✓ From cache"))

            # Source indicators (non-cached)
            else:
                source_info = response_data.get("source_info", {})
                source_type = source_info.get("type", "direct")

                if source_type == "rag":
                    elements.append(cl.Text(name="rag", content="✓ Used knowledge base"))
                elif source_type == "rag_web":
                    elements.append(cl.Text(name="rag_web", content="✓ Used knowledge base + web search"))

            await cl.Message(content=response_text, elements=elements).send()

        else:
            # Handle API error
            error_message = f"Something went wrong: {response.status_code}"
            try:
                error_data = response.json()
                error_message += f" - {error_data.get('message', '')}"
            except:
                error_message += f" - {response.text}"

            await cl.Message(content=error_message).send()

    except requests.exceptions.Timeout:
        await cl.Message(content="API request timeout. Please retry or check server status.").send()
    except requests.exceptions.ConnectionError:
        await cl.Message(content="Cannot connect to backend service. Please ensure server is running.").send()
    except Exception as e:
        error_message = f"Error occurred: {str(e)}"
        print(f"Detailed error: {traceback.format_exc()}")
        await cl.Message(content=error_message).send()



async def process_file(file):
    """Process uploaded file and send to Django API"""
    temp_file_path = None

    try:
        # Validate file object
        if not hasattr(file, 'name'):
            print(f"Skipping non-file element: {type(file)}")
            return

        print(f"Processing file: {file.name}")

        # Show processing message
        processing_msg = cl.Message(content=f"Processing your uploaded file: {file.name}...")
        await processing_msg.send()

        # Check file type
        file_ext = os.path.splitext(file.name)[1].lower()
        image_formats = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        is_image = file_ext in image_formats
        supported_formats = ['.pdf', '.txt', '.docx', '.doc'] + image_formats

        if file_ext not in supported_formats:
            await cl.Message(
                content=f"Unsupported file type: {file_ext}. Please upload PDF, TXT, Word, or image files.").send()
            return

        # Create temp directory
        temp_dir = os.path.join(project_root, "data", "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Generate safe filename
        import re
        safe_filename = re.sub(r'[\\/*?:"<>|]', '_', file.name)
        temp_file_path = os.path.join(temp_dir, safe_filename)

        # Save file using primary method (get_bytes)
        # 保存上传的文件，兼容老版本 Chainlit
        file_saved = False

        # content
        if not file_saved and hasattr(file, 'content') and file.content:
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(file.content)
                file_saved = True
                print("File saved successfully using content attribute")
            except Exception as e:
                print(f"content attribute method failed: {e}")

        # if not file_saved and hasattr(file, 'path') and os.path.exists(file.path):
        #     try:
        #         shutil.copy(file.path, temp_file_path)
        #         file_saved = True
        #         print("File saved successfully using path attribute")
        #     except Exception as e:
        #         print(f"path attribute method failed: {e}")

        # all fail
        if not file_saved:
            print("All file saving methods failed")
            await cl.Message(content="Unable to read file content. Please try another file.").send()
            return

        # Upload to Django API
        await upload_to_django_api(temp_file_path, safe_filename, is_image)

    except Exception as e:
        error_message = f"Error processing file: {str(e)}"
        print(f"Detailed error: {traceback.format_exc()}")
        await cl.Message(content=error_message).send()
    finally:
        # Cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Deleted temp file: {temp_file_path}")
            except Exception as e:
                print(f"Error deleting temp file: {e}")


import aiohttp
import asyncio

async def upload_to_django_api(file_path, filename, is_image):
    """Upload file to Django API - async version"""
    try:
        # Choose API endpoint based on file type
        if is_image:
            upload_url = f"{DJANGO_API_BASE_URL}/analyze_image/"
            print(f"Uploading image {filename} to analysis API: {upload_url}")
        else:
            upload_url = f"{DJANGO_API_BASE_URL}/upload/"
            print(f"Uploading file {filename} to Django API: {upload_url}")

        # Use async HTTP client
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                # Create form data
                data = aiohttp.FormData()
                data.add_field('file', f, filename=filename)

                print(f"Starting async file upload: {filename}")

                # Async upload to prevent blocking the event loop
                async with session.post(
                        upload_url,
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=300)  # 5-minute timeout
                ) as response:

                    print(f"response.status: {response.status}")

                    if response.status == 200:
                        response_data = await response.json()
                        success_message = response_data.get("message", "File processed successfully")

                        print(f"Preparing to send success message: {success_message}")
                        await cl.Message(content=success_message).send()
                        print("Success message sent")

                        # Show image analysis results if available
                        if is_image and "analysis" in response_data:
                            analysis_result = response_data.get("analysis", "")
                            await cl.Message(content=f"Image Analysis Results:\n\n{analysis_result}").send()

                        # Enable RAG mode and show reminder
                        cl.user_session.set("use_rag", True)
                        await cl.Message(
                            content="I can now answer questions based on this file. Feel free to ask!",
                            elements=[
                                cl.Text(name="rag_reminder",
                                        content="Tip: Use `/rag off` or `/rag on` to toggle knowledge base reference.")
                            ]
                        ).send()
                        print("All messages sent")
                    else:
                        response_text = await response.text()
                        error_message = f"File upload failed: {response.status}"
                        try:
                            error_data = await response.json()
                            error_message += f" - {error_data.get('message', '')}"
                        except:
                            error_message += f" - {response_text}"
                        await cl.Message(content=error_message).send()

    except asyncio.TimeoutError:
        await cl.Message(content="File upload timeout. File might be too large or server overloaded.").send()
    except aiohttp.ClientConnectionError:
        await cl.Message(content="Cannot connect to backend service. Please ensure server is running.").send()
    except Exception as e:
        error_message = f"Error uploading file: {str(e)}"
        print(f"Detailed error: {traceback.format_exc()}")
        await cl.Message(content=error_message).send()

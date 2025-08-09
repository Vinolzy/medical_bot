#!/usr/bin/env python
"""
Startup Script
Launch Django backend and Chainlit frontend
"""
import os
import sys
import subprocess
import requests
import time
import webbrowser
from threading import Thread

# URL & Port Settings
DJANGO_HOST = "localhost"
DJANGO_PORT = "8007"
CHAINLIT_PORT = "8888"


def run_django_server():
    """Start Django server"""
    print("Starting Django backend service...")
    # Set environment variable: find settings module in the package named medical_bot
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "medical_bot.settings")

    # manage.py is a Python script automatically generated when creating each Django project
    # It's a command-line tool containing many commands for managing and controlling Django projects
    # Execute Django migrate command: update or create database table structure
    print("Django migrate")
    subprocess.run([sys.executable, "manage.py", "migrate", "--noinput"], check=False)

    # Start Django development server
    subprocess.run([sys.executable, "manage.py", "runserver", f"{DJANGO_HOST}:{DJANGO_PORT}"])


def run_chainlit_app():
    """Start Chainlit application"""
    os.chdir("chainlit_app")
    subprocess.run([sys.executable, "-m", "chainlit", "run", "app.py", "--port", CHAINLIT_PORT])


def open_browser():
    webbrowser.open(f"http://{DJANGO_HOST}:{CHAINLIT_PORT}")


def main():
    """Main function"""
    # Set Django API base URL environment variable
    os.environ["DJANGO_API_BASE_URL"] = f"http://{DJANGO_HOST}:{DJANGO_PORT}/api/v1"

    # Ensure necessary directories exist
    os.makedirs("data/knowledge_base", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)  # Ensure cache directory exists

    # Create and start Django server thread
    django_thread = Thread(target=run_django_server)
    django_thread.daemon = True  # Main program won't wait for daemon thread to finish before exiting
    django_thread.start()

    # Wait for Django server to start
    django_connected = False

    while not django_connected:
        time.sleep(2)
        try:
            response = requests.get(f"http://{DJANGO_HOST}:{DJANGO_PORT}/api/v1/health/", timeout=3)
            if response.status_code == 200:  # Request successful
                django_connected = True
        except Exception as e:
            print("Waiting for Django...")

    print("Chainlit running...")
    run_chainlit_app()


if __name__ == "__main__":
    main()
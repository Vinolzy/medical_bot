"""
Django URL configuration
"""
from django.urls import path
from medical_bot import views

# API version prefix
API_PREFIX = "api/v1"

urlpatterns = [
    # Base endpoint
    path('', views.index, name='index'),

    # API endpoints
    path(f'{API_PREFIX}/medicalbot/', views.medicalbot_api, name='medicalbot_api'),
    path(f'{API_PREFIX}/health/', views.health_check, name='health_check'),
    path(f'{API_PREFIX}/upload/', views.upload_file, name='upload_file'),
    path(f'{API_PREFIX}/rebuild_kb/', views.rebuild_knowledge_base, name='rebuild_kb'),

    # Cache-related endpoints
    path(f'{API_PREFIX}/cache_stats/', views.cache_stats, name='cache_stats'),
    path(f'{API_PREFIX}/clear_cache/', views.clear_cache, name='clear_cache'),

    # Speech-related endpoints
    path(f'{API_PREFIX}/speech_to_text/', views.speech_to_text, name='speech_to_text'),
    path(f'{API_PREFIX}/text_to_speech/', views.text_to_speech, name='text_to_speech'),

    # Vision-related endpoints
    path(f'{API_PREFIX}/analyze_image/', views.analyze_image, name='analyze_image'),
]

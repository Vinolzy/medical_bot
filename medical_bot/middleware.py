"""
Django middleware
"""

def cors_middleware(get_response):
    """CORS middleware"""
    def middleware(request):
        response = get_response(request)
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        return response
    return middleware
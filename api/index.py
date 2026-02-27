def handler(request):
    """aether v1.0 API endpoint"""
    import json
    from urllib.parse import parse_qs, urlparse
    
    if request.method == "GET":
        path = urlparse(request.url).path
        
        if path == "/" or path == "":
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "name": "aether",
                    "version": "1.0.0",
                    "status": "ok"
                }),
                "headers": {"Content-Type": "application/json"}
            }
        elif path == "/health":
            return {
                "statusCode": 200,
                "body": json.dumps({"status": "ok"}),
                "headers": {"Content-Type": "application/json"}
            }
        elif path == "/info":
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "name": "aether",
                    "version": "1.0.0",
                    "params": "3.5M",
                    "github": "https://github.com/nulljosh/aether"
                }),
                "headers": {"Content-Type": "application/json"}
            }
    
    return {
        "statusCode": 404,
        "body": json.dumps({"error": "Not found"}),
        "headers": {"Content-Type": "application/json"}
    }

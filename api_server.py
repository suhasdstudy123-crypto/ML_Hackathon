"""Backward-compatible Uvicorn entry point. All production logic lives in backend.app."""
from backend.app.main import app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api_server:app', host='0.0.0.0', port=8000)

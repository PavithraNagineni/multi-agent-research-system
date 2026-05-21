import uvicorn
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from api.app import app

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

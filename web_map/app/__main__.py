"""python -m app [--port 8000] [--host 127.0.0.1]"""
import argparse

import uvicorn

parser = argparse.ArgumentParser(description="Run the Ghana EUDR tree-crop web map.")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--reload", action="store_true", help="restart on code changes (development)")
args = parser.parse_args()

uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)

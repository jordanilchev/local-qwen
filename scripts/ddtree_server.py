#!/usr/bin/env python3
"""
OpenAI-compatible chat server backed by DDTree + Qwen3.5-27B-4bit.

The canonical implementation lives in vendor/ddtree-mlx/ddtree_server.py.
This script is the project-level entry point with default settings baked in.

Usage:
    HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/ddtree_server.py
    HF_HOME=~/Models/HuggingFace .venv/bin/python scripts/ddtree_server.py --port 8006

Endpoints:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions
"""
import os, sys
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Models/HuggingFace"))

# Ensure the vendored source is importable even if not installed editable
_vendor = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ddtree-mlx")
if _vendor not in sys.path:
    sys.path.insert(0, _vendor)

# Delegate entirely to the vendored server entrypoint
import runpy
runpy.run_path(os.path.join(_vendor, "ddtree_server.py"), run_name="__main__")

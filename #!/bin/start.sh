#!/bin/bash
uvicorn api.caption:app --host 0.0.0.0 --port ${PORT:-8000}

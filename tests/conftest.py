# tests/conftest.py
import os
import sys

# pytest-dotenv has already loaded .env at this point
pp = os.getenv("PYTHONPATH")
if pp:
    for p in pp.split(os.pathsep):
        if p:
            sys.path.insert(0, p)

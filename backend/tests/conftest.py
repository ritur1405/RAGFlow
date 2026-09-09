"""Shared test fixtures.

Sets dummy environment variables so that module-level code in
services.py and database.py does not crash during test collection.
These are set before any app module is imported.
"""

import os

# Must be set before any app imports trigger module-level initialization
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")

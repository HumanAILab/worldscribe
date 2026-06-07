"""Centralized, environment-driven configuration for secrets and project-specific
identifiers (Firebase credentials, database URL, etc.).

Values are loaded from environment variables (and a local, gitignored `.env` file
if `python-dotenv` is installed). Never hardcode credentials, file names, URLs, or
IP addresses in source files; add them to `.env` instead.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Path to the Firebase Admin SDK service-account JSON file (kept out of git).
FIREBASE_CREDENTIALS = os.environ.get("FIREBASE_CREDENTIALS", "firebase-credentials.json")

# Firebase Realtime Database URL, e.g. https://<project>-default-rtdb.firebaseio.com/
FIREBASE_DATABASE_URL = os.environ.get("FIREBASE_DATABASE_URL", "")

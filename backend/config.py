import os
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SARVAM_API_KEY or not GROQ_API_KEY:
    print("Warning: SARVAM_API_KEY or GROQ_API_KEY is not set in .env file.")

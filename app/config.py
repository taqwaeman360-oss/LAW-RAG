import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATA_DIR = os.getenv("DATA_DIR", "./data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

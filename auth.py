from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env

HF_TOKEN = os.getenv("HF_TOKEN")

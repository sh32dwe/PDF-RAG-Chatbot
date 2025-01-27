import os
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    return os.getenv('HUGGINGFACE_TOKEN')

def validate_pdf_path(pdf_path):
    """Validate if PDF file exists and is accessible"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("File must be a PDF")
    return True
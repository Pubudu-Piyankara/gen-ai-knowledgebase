# Fix the validate_request_data function for FastAPI
import logging
import os
import config
from config import Config

logger = logging.getLogger(__name__)

config = Config()

async def validate_request_data(form_data: dict, files: dict):
    """Validate incoming request data."""
    required_fields = ['type', 'fileName', 'fileType', 'spaceId', 'storageMethod', 'memoryId']
    
    for field in required_fields:
        if field not in form_data or not form_data[field]:
            raise ValueError(f"Missing required field: {field}")
    
    if 'file' not in files:
        raise ValueError("No file uploaded")
    
    if form_data['type'] != "upload":
        raise ValueError("Invalid type, must be 'upload'")
    
    if form_data['storageMethod'] not in ["storage", "database"]:
        raise ValueError("Invalid storage method")
 
 # Helper function to validate file
def validate_file(temp_path: str, file_name: str) -> int:
    """Validate file size and type."""
    # Get file size
    file_size = os.path.getsize(temp_path)
    
    # Validate file size (this is already handled by Flask's MAX_CONTENT_LENGTH, but double-check)
    if file_size > config.MAX_CONTENT_LENGTH:
        raise ValueError("File too large")
    
    # Validate file type
    if hasattr(config, 'document_processor') and hasattr(config.document_processor, 'is_allowed_file'):
        if not config.document_processor.is_allowed_file(file_name):
            raise ValueError(f"File type not supported. Allowed types: {config.document_processor.allowed_extensions}")
    
    logger.info(f"File validation passed - Size: {file_size} bytes, Name: {file_name}")
    return file_size
 
def validate_file_size(file_content: bytes, max_size: int) -> None:
    """
    Validate that file size does not exceed maximum allowed.
    
    Args:
        file_content: Binary file content
        max_size: Maximum allowed size in bytes
    
    Raises:
        ValueError: If file size exceeds maximum
    """
    file_size = len(file_content)
    
    if file_size > max_size:
        logger.error(f"File size {file_size} bytes exceeds maximum {max_size} bytes")
        raise ValueError(f"File too large: {file_size} bytes (max: {max_size} bytes)")
    
    logger.info(f"File size validation passed: {file_size} bytes")

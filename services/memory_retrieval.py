"""
Memory Retrieval Service
Handles file content retrieval from various sources (Supabase storage, base64 data)
"""

import base64
import logging
import requests
from typing import Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def retrieve_file_content(file_url: str, storage_method: str) -> bytes:
    """
    Retrieve file content based on storage method.
    
    Args:
        file_url: URL or base64 data string
        storage_method: Either 'storage' (download from URL) or 'database' (decode base64)
    
    Returns:
        bytes: Raw binary file content
    
    Raises:
        requests.exceptions.HTTPError: If download fails
        requests.exceptions.RequestException: If network error occurs
        ValueError: If base64 decoding fails or invalid storage method
    """
    
    if storage_method == 'storage':
        return _download_file_from_url(file_url)
    elif storage_method == 'database':
        return _decode_base64_content(file_url)
    else:
        raise ValueError(f"Invalid storage method: {storage_method}. Must be 'storage' or 'database'")

def _download_file_from_url(url: str) -> bytes:
    """
    Download file from a public URL (e.g., Supabase storage).
    
    Args:
        url: Public URL to download from
    
    Returns:
        bytes: Downloaded file content
    
    Raises:
        requests.exceptions.HTTPError: If HTTP error occurs (4xx, 5xx)
        requests.exceptions.RequestException: If network error occurs
    """
    try:
        logger.info(f"Downloading file from URL: {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        file_size = len(response.content)
        logger.info(f"Successfully downloaded file. Size: {file_size} bytes")
        
        return response.content
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error downloading file: {str(e)}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading file: {str(e)}")
        raise

def _decode_base64_content(base64_data: str) -> bytes:
    """
    Decode base64 encoded file content.
    
    Args:
        base64_data: Base64 encoded string (may include data URL prefix)
    
    Returns:
        bytes: Decoded binary content
    
    Raises:
        ValueError: If base64 decoding fails
    """
    try:
        logger.info("Decoding base64 file content")
        
        # Remove data URL prefix if present (e.g., "data:image/png;base64,")
        if ',' in base64_data:
            base64_data = base64_data.split(',', 1)[1]
        
        decoded_content = base64.b64decode(base64_data)
        file_size = len(decoded_content)
        
        logger.info(f"Successfully decoded base64 content. Size: {file_size} bytes")
        
        return decoded_content
        
    except Exception as e:
        logger.error(f"Error decoding base64 data: {str(e)}")
        raise ValueError(f"Failed to decode base64 content: {str(e)}")

def get_file_mime_type(file_name: str) -> str:
    """
    Determine MIME type based on file extension.
    
    Args:
        file_name: Name of the file including extension
    
    Returns:
        str: MIME type (e.g., 'application/pdf', 'image/jpeg')
    """
    
    # MIME type mapping
    mime_type_map = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xls': 'application/vnd.ms-excel',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.ppt': 'application/vnd.ms-powerpoint',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.mp3': 'audio/mpeg',
        '.mp4': 'video/mp4',
        '.wav': 'audio/wav',
        '.json': 'application/json',
        '.xml': 'application/xml',
        '.csv': 'text/csv',
        '.html': 'text/html',
        '.htm': 'text/html'
    }
    
    file_extension = Path(file_name).suffix.lower()
    mime_type = mime_type_map.get(file_extension, 'application/octet-stream')
    
    logger.debug(f"File: {file_name}, Extension: {file_extension}, MIME type: {mime_type}")
    
    return mime_type

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

def prepare_multipart_data(
    file_content: bytes,
    file_name: str,
    memory_id: str,
    space_id: str,
    storage_method: str
) -> Tuple[dict, dict]:
    """
    Prepare multipart form data for internal API forwarding.
    
    Args:
        file_content: Binary file content
        file_name: Original file name
        memory_id: Memory UUID
        space_id: Space UUID
        storage_method: Storage method used
    
    Returns:
        Tuple of (files dict, data dict) for requests.post
    """
    import io
    
    mime_type = get_file_mime_type(file_name)
    file_size = len(file_content)
    
    files = {
        'file': (file_name, io.BytesIO(file_content), mime_type)
    }
    
    data = {
        'type': 'upload',
        'fileName': file_name,
        'fileType': mime_type,
        'fileSize': str(file_size),
        'spaceId': space_id,
        'storageMethod': storage_method,
        'memoryId': memory_id
    }
    
    logger.debug(f"Prepared multipart data - File: {file_name}, Size: {file_size}, MIME: {mime_type}")
    
    return files, data
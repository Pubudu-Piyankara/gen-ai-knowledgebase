"""
Memory Retrieval Service
Handles file content retrieval from various sources (Supabase storage, base64 data)
"""

import base64
import logging
import requests
import os
from typing import Tuple, Optional
from pathlib import Path
from supabase import create_client, Client
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from urllib.parse import urlparse

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


def retrieve_file_from_supabase_s3(
    file_url: str,
    s3_endpoint: str,
    region: str,
    access_key_id: str,
    secret_access_key: str,
    bucket_name: str
) -> bytes:
    """
    Retrieve file from Supabase private storage using S3-compatible API.
    
    Args:
        file_url: Full Supabase storage URL
        s3_endpoint: S3-compatible endpoint URL
        region: S3 region
        access_key_id: S3 access key ID
        secret_access_key: S3 secret access key
        bucket_name: S3 bucket name
    
    Returns:
        bytes: Downloaded file content
        
    Raises:
        ValueError: If download fails or file not found
    """
    try:
        logger.info("Attempting S3-compatible download from Supabase private storage...")
        
        # Parse the file path from the URL
        bucket, file_path = _parse_supabase_url(file_url, bucket_name)
        
        if not file_path:
            raise ValueError(f"Could not extract file path from URL: {file_url}")
        
        logger.info(f"S3 Download - Bucket: {bucket}, Path: {file_path}")
        logger.info(f"Using endpoint: {s3_endpoint}")
        
        # Initialize S3 client with Supabase credentials
        s3_client = boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key
        )
        
        # Download the file
        response = s3_client.get_object(Bucket=bucket, Key=file_path)
        file_content = response['Body'].read()
        
        if len(file_content) > 0:
            logger.info(f"✅ Successfully downloaded via S3 API, size: {len(file_content)} bytes")
            return file_content
        else:
            raise ValueError("Downloaded file is empty")
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.error(f"File not found in S3: {file_path}")
            raise ValueError(f"File not found: {file_path}")
        elif error_code == 'AccessDenied':
            logger.error(f"Access denied to S3 file: {file_path}")
            raise ValueError(f"Access denied: {file_path}")
        else:
            logger.error(f"S3 ClientError: {str(e)}")
            raise ValueError(f"S3 error: {str(e)}")
            
    except NoCredentialsError:
        logger.error("S3 credentials not available")
        raise ValueError("S3 credentials not configured")
        
    except Exception as e:
        logger.error(f"Unexpected error in S3 download: {str(e)}")
        raise ValueError(f"S3 download failed: {str(e)}")

def retrieve_file_from_supabase_advanced(
    file_url: str,
    supabase_url: str, 
    supabase_key: str,
    bucket_name: Optional[str] = None,
    s3_config: Optional[dict] = None
) -> bytes:
    """
    Advanced file retrieval from Supabase with S3 and fallback methods.
    
    Args:
        file_url: Full Supabase storage URL
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        bucket_name: Optional bucket name override
        s3_config: S3 configuration dict with keys: endpoint, region, access_key_id, secret_access_key
    
    Returns:
        bytes: Downloaded file content
        
    Raises:
        ValueError: If all download methods fail
    """
    download_success = False
    file_content = None
    
    # METHOD 1: Try S3-compatible API first (for private storage)
    if s3_config and all(k in s3_config for k in ['endpoint', 'region', 'access_key_id', 'secret_access_key']):
        try:
            logger.info("Method 1: Attempting S3-compatible download...")
            
            file_content = retrieve_file_from_supabase_s3(
                file_url=file_url,
                s3_endpoint=s3_config['endpoint'],
                region=s3_config['region'],
                access_key_id=s3_config['access_key_id'],
                secret_access_key=s3_config['secret_access_key'],
                bucket_name=bucket_name or s3_config.get('bucket', 'uploads')
            )
            
            if file_content and len(file_content) > 0:
                download_success = True
                
        except Exception as s3_error:
            logger.warning(f"S3 download failed: {str(s3_error)}")
    
    # METHOD 2: Try Supabase Storage API (fallback)
    if not download_success:
        try:
            logger.info("Method 2: Attempting Supabase Storage API...")
            
            supabase = create_client(supabase_url, supabase_key)
            bucket, file_path = _parse_supabase_url(file_url, bucket_name)
            
            if bucket and file_path:
                file_content = supabase.storage.from_(bucket).download(file_path)
                
                if file_content and len(file_content) > 0:
                    logger.info(f"✅ Successfully downloaded via Supabase API, size: {len(file_content)} bytes")
                    download_success = True
                    
        except Exception as api_error:
            logger.warning(f"Supabase API failed: {str(api_error)}")
    
    # METHOD 3: Try signed URL (fallback)
    if not download_success:
        try:
            logger.info("Method 3: Attempting signed URL...")
            
            supabase = create_client(supabase_url, supabase_key)
            bucket, file_path = _parse_supabase_url(file_url, bucket_name)
            
            if bucket and file_path:
                signed_url_response = supabase.storage.from_(bucket).create_signed_url(file_path, 3600)
                
                if signed_url_response and 'signedURL' in signed_url_response:
                    signed_url = signed_url_response['signedURL']
                    response = requests.get(signed_url, timeout=30)
                    response.raise_for_status()
                    file_content = response.content
                    
                    if len(file_content) > 0:
                        logger.info(f"✅ Successfully downloaded via signed URL, size: {len(file_content)} bytes")
                        download_success = True
                        
        except Exception as signed_error:
            logger.warning(f"Signed URL download failed: {str(signed_error)}")
    
    # METHOD 4: Debug - List bucket contents
    if not download_success:
        try:
            logger.info("Method 4: Debug listing bucket contents...")
            
            if s3_config:
                s3_client = boto3.client(
                    's3',
                    endpoint_url=s3_config['endpoint'],
                    region_name=s3_config['region'],
                    aws_access_key_id=s3_config['access_key_id'],
                    aws_secret_access_key=s3_config['secret_access_key']
                )
                
                bucket, file_path = _parse_supabase_url(file_url, bucket_name)
                
                # List bucket contents
                response = s3_client.list_objects_v2(Bucket=bucket, Prefix='')
                
                if 'Contents' in response:
                    logger.info("Bucket contents:")
                    for obj in response['Contents'][:10]:  # Show first 10 objects
                        logger.info(f"  - {obj['Key']}")
                    
                    # Check if our file exists with different path
                    target_filename = file_path.split('/')[-1]
                    for obj in response['Contents']:
                        if obj['Key'].endswith(target_filename):
                            logger.info(f"Found file with key: {obj['Key']}")
                            # Try downloading with the found key
                            response = s3_client.get_object(Bucket=bucket, Key=obj['Key'])
                            file_content = response['Body'].read()
                            
                            if len(file_content) > 0:
                                logger.info(f"✅ Successfully downloaded with corrected path, size: {len(file_content)} bytes")
                                download_success = True
                                break
                else:
                    logger.warning("Bucket appears to be empty or inaccessible")
                    
        except Exception as debug_error:
            logger.warning(f"Debug listing failed: {str(debug_error)}")
    
    if not download_success or not file_content:
        bucket, file_path = _parse_supabase_url(file_url, bucket_name)
        error_details = f"""
        All download methods failed for URL: {file_url}
        
        Parsed details:
        - Bucket: {bucket}
        - File path: {file_path}
        - S3 endpoint: {s3_config.get('endpoint') if s3_config else 'Not configured'}
        
        Troubleshooting steps:
        1. Verify the file exists at the exact path
        2. Check S3 credentials have proper permissions
        3. Ensure the bucket name is correct
        4. Verify the file path structure matches uploaded file
        """
        
        logger.error(error_details)
        raise ValueError(error_details)
    
    return file_content

def retrieve_file_from_supabase_storage(
    supabase_url: str,
    supabase_key: str,
    bucket_name: str,
    file_path: str,
    destination_path: Optional[str] = None
) -> Tuple[bytes, str]:
    """
    Retrieve a file from Supabase Storage using bucket and file path.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        bucket_name: Storage bucket name
        file_path: Path to file within bucket
        destination_path: Optional local path to save file
    
    Returns:
        Tuple of (file_content, file_name)
        
    Raises:
        ValueError: If file not found or download fails
        Exception: If Supabase client initialization fails
    """
    try:
        # Initialize Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        logger.info(f"Initialized Supabase client for bucket: {bucket_name}")

        # Get file name from path
        file_name = os.path.basename(file_path)

        # Download file from Supabase Storage
        response = supabase.storage.from_(bucket_name).download(file_path)

        if response is None or len(response) == 0:
            logger.error(f"File not found or empty: {file_path} in bucket {bucket_name}")
            raise ValueError(f"File {file_path} not found in bucket {bucket_name}")

        # If destination_path is provided, save the file locally
        if destination_path:
            try:
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                with open(destination_path, 'wb') as f:
                    f.write(response)
                logger.info(f"File saved to {destination_path}")
            except Exception as e:
                logger.error(f"Failed to save file to {destination_path}: {str(e)}")
                raise ValueError(f"Failed to save file: {str(e)}")

        logger.info(f"Successfully retrieved file: {file_path} from bucket {bucket_name}, size: {len(response)} bytes")
        return response, file_name

    except Exception as e:
        logger.error(f"Error retrieving file from Supabase: {str(e)}")
        raise ValueError(f"Failed to retrieve file: {str(e)}")

def _parse_supabase_url(file_url: str, bucket_override: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse Supabase storage URL to extract bucket and file path.
    Handles both public and private storage URLs.
    
    Args:
        file_url: Full Supabase storage URL
        bucket_override: Optional bucket name to use instead of parsed one
        
    Returns:
        Tuple of (bucket_name, file_path) or (None, None) if parsing fails
    """
    try:
        logger.info(f"Parsing Supabase URL: {file_url}")
        
        # Handle different URL formats
        # Format 1: https://PROJECT.supabase.co/storage/v1/object/public/BUCKET/PATH
        # Format 2: https://PROJECT.supabase.co/storage/v1/object/BUCKET/PATH (for private)
        
        if '/storage/v1/object/public/' in file_url:
            # Public storage URL
            url_parts = file_url.split('/storage/v1/object/public/')
            if len(url_parts) == 2:
                path_parts = url_parts[1].split('/', 1)
                parsed_bucket = bucket_override or path_parts[0]
                file_path = path_parts[1] if len(path_parts) > 1 else ''
                logger.info(f"Parsed public URL - Bucket: {parsed_bucket}, Path: {file_path}")
                return parsed_bucket, file_path
                
        elif '/storage/v1/object/' in file_url:
            # Private storage URL (without 'public')
            url_parts = file_url.split('/storage/v1/object/')
            if len(url_parts) == 2:
                path_parts = url_parts[1].split('/', 1)
                parsed_bucket = bucket_override or path_parts[0]
                file_path = path_parts[1] if len(path_parts) > 1 else ''
                logger.info(f"Parsed private URL - Bucket: {parsed_bucket}, Path: {file_path}")
                return parsed_bucket, file_path
        
        # If we have a bucket override, try to extract just the file path
        if bucket_override:
            # Try to find the path after the bucket name
            if f'/{bucket_override}/' in file_url:
                path_start = file_url.find(f'/{bucket_override}/') + len(f'/{bucket_override}/')
                file_path = file_url[path_start:]
                logger.info(f"Using bucket override - Bucket: {bucket_override}, Path: {file_path}")
                return bucket_override, file_path
        
        logger.warning(f"Could not parse Supabase URL format: {file_url}")
        return None, None
            
    except Exception as e:
        logger.error(f"Error parsing Supabase URL: {str(e)}")
        return None, None
    
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

def create_signed_url(
    supabase_url: str,
    supabase_key: str,
    bucket_name: str,
    file_path: str,
    expires_in: int = 3600
) -> Optional[str]:
    """
    Create a signed URL for temporary access to a Supabase storage file.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        bucket_name: Storage bucket name
        file_path: Path to file within bucket
        expires_in: URL expiration time in seconds (default: 1 hour)
    
    Returns:
        Signed URL string or None if creation fails
    """
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        signed_url = supabase.storage.from_(bucket_name).create_signed_url(file_path, expires_in)
        
        logger.info(f"Created signed URL for {file_path}, expires in {expires_in} seconds")
        return signed_url
        
    except Exception as e:
        logger.error(f"Failed to create signed URL: {str(e)}")
        return None
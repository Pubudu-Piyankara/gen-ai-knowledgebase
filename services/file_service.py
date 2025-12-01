import os
import tempfile
from fastapi import HTTPException, Path, UploadFile
from fastapi.concurrency import run_in_threadpool
import httpx
from werkzeug.utils import secure_filename
import aiofiles
import logging
import re
from typing import List, Optional, Tuple
from config import Config
from supabase import create_client, Client
import pathlib 

logger = logging.getLogger(__name__)

config = Config()

async def save_uploaded_file(uploaded_file: UploadFile, original_filename: str) -> tuple[str, int]:
    """Save uploaded file to temporary location."""
    try:
        # Secure the filename
        filename = secure_filename(original_filename)
        
        # Create temporary file with proper extension
        temp_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(filename).suffix,
            prefix="upload_"
        ).name
        
        # Save the uploaded file (FastAPI way)
        with open(temp_path, "wb") as buffer:
            content = await uploaded_file.read()
            buffer.write(content)
        
        # Get file size
        file_size = os.path.getsize(temp_path)
        
        logger.info(f"File saved to temporary path: {temp_path}, Size: {file_size} bytes")
        return temp_path, file_size
        
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {str(e)}")
        raise ValueError(f"Failed to save uploaded file: {str(e)}")

def cleanup_temp_file(temp_path: str) -> None:
    """Safely cleanup temporary file."""
    try:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info(f"Cleaned up temporary file: {temp_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {temp_path}: {str(e)}")

def parse_supabase_storage_path(file_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse Supabase storage URL to extract bucket and file path.
    
    Examples:
    - https://xxx.supabase.co/storage/v1/object/public/uploads/path/to/file.docx
    - https://xxx.supabase.co/storage/v1/object/signed/bucket/path/to/file.docx
    
    Returns: (bucket_name, file_path)
    """
    try:
        # Handle public URLs
        if '/storage/v1/object/public/' in file_url:
            path_part = file_url.split('/storage/v1/object/public/')[1]
        # Handle signed URLs  
        elif '/storage/v1/object/signed/' in file_url:
            path_part = file_url.split('/storage/v1/object/signed/')[1]
        else:
            logger.error(f"Unrecognized Supabase storage URL format: {file_url}")
            return None, None
        
        # Split path into components
        path_components = path_part.split('/')
        
        if len(path_components) < 2:
            logger.error(f"Invalid path structure: {path_part}")
            return None, None
        
        # For private buckets, the structure is usually:
        # uploads/user_id/space_id/filename.ext
        # So we need the full path after the bucket name
        
        bucket_name = path_components[0]  # First part is bucket name
        file_path = '/'.join(path_components[1:])  # Rest is file path
        
        logger.info(f"Parsed Supabase URL - Bucket: {bucket_name}, Path: {file_path}")
        return bucket_name, file_path
        
    except Exception as e:
        logger.error(f"Error parsing Supabase URL: {str(e)}")
        return None, None

async def download_from_supabase_private(bucket_name: str, file_path: str) -> bytes:
    """
    Download file from private Supabase storage using authenticated client.
    """
    def _download_file():
        try:
            # Create Supabase client with service role key for better permissions
            supabase_key = config.SUPABASE_SERVICE_ROLE_KEY or config.SUPABASE_KEY
            supabase = create_client(config.SUPABASE_URL, supabase_key)
            
            logger.info(f"🔐 Downloading from private bucket: {bucket_name}/{file_path}")
            
            # Download file using Supabase client
            response = supabase.storage.from_(bucket_name).download(file_path)
            
            if response and len(response) > 0:
                logger.info(f"✅ Successfully downloaded from Supabase client: {len(response)} bytes")
                return response
            else:
                logger.error("❌ Supabase client returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Supabase client download failed: {str(e)}")
            # Try to list bucket contents for debugging
            try:
                files = supabase.storage.from_(bucket_name).list(path='')
                logger.info(f"📂 Bucket '{bucket_name}' contains {len(files)} items:")
                for f in files[:5]:  # Show first 5 files
                    logger.info(f"  - {f.get('name', 'unknown')}")
                    
                # Try to find the file with a different path
                target_filename = file_path.split('/')[-1]
                logger.info(f"🔍 Looking for file: {target_filename}")
                
                # List files recursively if possible
                def find_file_recursive(current_path=''):
                    try:
                        items = supabase.storage.from_(bucket_name).list(path=current_path)
                        for item in items:
                            item_name = item.get('name', '')
                            if item_name == target_filename:
                                found_path = f"{current_path}/{item_name}" if current_path else item_name
                                logger.info(f"🎯 Found file at: {found_path}")
                                return supabase.storage.from_(bucket_name).download(found_path)
                            elif item.get('metadata', {}).get('mimetype') is None:  # Might be a folder
                                nested_path = f"{current_path}/{item_name}" if current_path else item_name
                                result = find_file_recursive(nested_path)
                                if result:
                                    return result
                    except:
                        pass
                    return None
                
                alternate_content = find_file_recursive()
                if alternate_content:
                    logger.info(f"✅ Found file via recursive search: {len(alternate_content)} bytes")
                    return alternate_content
                    
            except Exception as debug_error:
                logger.warning(f"Debug listing failed: {debug_error}")
            
            raise e
    
    # Run the download in a thread pool to avoid blocking
    return await run_in_threadpool(_download_file)

async def download_with_signed_url(bucket_name: str, file_path: str) -> bytes:
    """
    Download file using Supabase signed URL method.
    """
    def _get_signed_url_and_download():
        try:
            # Create Supabase client
            supabase_key = config.SUPABASE_SERVICE_ROLE_KEY or config.SUPABASE_KEY
            supabase = create_client(config.SUPABASE_URL, supabase_key)
            
            logger.info(f"🔗 Creating signed URL for: {bucket_name}/{file_path}")
            
            # Create signed URL (valid for 1 hour)
            signed_url_response = supabase.storage.from_(bucket_name).create_signed_url(file_path, 3600)
            
            if not signed_url_response or 'signedURL' not in signed_url_response:
                logger.error("❌ Failed to create signed URL")
                return None
            
            signed_url = signed_url_response['signedURL']
            logger.info(f"✅ Created signed URL: {signed_url[:100]}...")
            
            # Download using the signed URL
            import requests
            response = requests.get(signed_url, timeout=60)
            response.raise_for_status()
            
            content = response.content
            logger.info(f"✅ Downloaded via signed URL: {len(content)} bytes")
            return content
            
        except Exception as e:
            logger.error(f"❌ Signed URL download failed: {str(e)}")
            raise e
    
    return await run_in_threadpool(_get_signed_url_and_download)

async def async_download_file(file_url: str, storage_method: str, bucket_name: str) -> bytes:
    """
    Async file download from Supabase storage with multiple fallback methods.
    Prioritizes private bucket authentication over public URLs.
    """
    
    try:
        logger.info(f"🚀 Starting download from: {file_url}")
        logger.info(f"Storage method: {storage_method}, Configured bucket: {bucket_name}")
        
        if storage_method == 'database':
            # Handle base64 data URLs
            if file_url.startswith('data:'):
                import base64
                try:
                    header, encoded = file_url.split(',', 1)
                    file_content = base64.b64decode(encoded)
                    logger.info(f"✅ Decoded base64 data, size: {len(file_content)} bytes")
                    return file_content
                except Exception as e:
                    logger.error(f"Failed to decode base64 data: {str(e)}")
                    raise HTTPException(status_code=400, detail="Invalid base64 data")
        
        # Handle Supabase storage URLs
        if 'supabase' in file_url.lower() and '/storage/v1/object/' in file_url:
            
            # Parse the URL to extract bucket and file path
            parsed_bucket, file_path = parse_supabase_storage_path(file_url)
            
            if not parsed_bucket or not file_path:
                raise HTTPException(status_code=400, detail="Could not parse Supabase storage URL")
            
            # Use the parsed bucket name, or fall back to configured bucket
            actual_bucket = parsed_bucket
            
            logger.info(f"🎯 Target: Bucket='{actual_bucket}', Path='{file_path}'")
            
            # METHOD 1: Try Supabase client direct download (for private buckets)
            try:
                logger.info("🔐 Method 1: Supabase client download (private bucket)")
                file_content = await download_from_supabase_private(actual_bucket, file_path)
                if file_content and len(file_content) > 0:
                    return file_content
            except Exception as e:
                logger.warning(f"Method 1 failed: {str(e)}")
            
            # METHOD 2: Try signed URL download
            try:
                logger.info("🔗 Method 2: Signed URL download")
                file_content = await download_with_signed_url(actual_bucket, file_path)
                if file_content and len(file_content) > 0:
                    return file_content
            except Exception as e:
                logger.warning(f"Method 2 failed: {str(e)}")
            
            # METHOD 3: Try public URL (as last resort)
            try:
                logger.info("🌐 Method 3: Public URL download (fallback)")
                async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                    headers = {'User-Agent': 'RAG-System/1.0'}
                    
                    # Try the original URL and some variations
                    urls_to_try = [
                        file_url,  # Original URL
                        file_url.replace('/object/public/', '/object/public/').replace('/uploads/', f'/{actual_bucket}/'),
                    ]
                    
                    for url in urls_to_try:
                        try:
                            logger.info(f"🔄 Trying public URL: {url}")
                            response = await client.get(url, headers=headers)
                            if response.status_code == 200:
                                content = response.content
                                if len(content) > 0:
                                    logger.info(f"✅ Success with public URL: {len(content)} bytes")
                                    return content
                        except Exception as url_error:
                            logger.warning(f"Public URL failed: {url_error}")
                            
            except Exception as e:
                logger.warning(f"Method 3 failed: {str(e)}")
            
            # If all methods failed
            raise HTTPException(
                status_code=404, 
                detail=f"File not found in bucket '{actual_bucket}' at path '{file_path}'. "
                       f"Tried private client, signed URL, and public URL methods."
            )
        
        # Handle non-Supabase URLs
        else:
            logger.info("🌐 Non-Supabase URL - using direct HTTP")
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                response = await client.get(file_url)
                response.raise_for_status()
                return response.content
            
    except HTTPException:
        # Re-raise our custom HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error downloading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

async def async_save_temp_file(file_content: bytes, filename: str) -> str:
    """Save file content to temporary file using aiofiles."""
    try:
        # Extract file extension safely
        file_extension = extract_file_extension(filename)
        
        # Create temporary file with proper extension
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension,
            prefix="ingest_",
            dir=tempfile.gettempdir()
        )
        temp_path = temp_file.name
        temp_file.close()  # Close the file handle
        
        # Write file asynchronously
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(file_content)
        
        logger.info(f"💾 Saved file to: {temp_path} (extension: {file_extension})")
        return temp_path
        
    except Exception as e:
        logger.error(f"Failed to save temporary file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
def extract_file_extension(filename: str) -> str:
    """
    Safely extract file extension from filename.
    Returns proper extension or .tmp as fallback.
    """
    try:
        # Remove any path components and get just the filename
        basename = os.path.basename(filename)
        
        # Use pathlib to get extension
        path_obj = pathlib.Path(basename)
        extension = path_obj.suffix.lower()
        
        # Validate extension
        valid_extensions = {'.docx', '.doc', '.pdf', '.txt', '.md', '.rtf', '.odt'}
        
        if extension in valid_extensions:
            logger.info(f"✅ Valid extension extracted: {extension}")
            return extension
        
        # Try to guess from filename if extension is missing or invalid
        basename_lower = basename.lower()
        if 'docx' in basename_lower:
            return '.docx'
        elif basename_lower.endswith('.doc'):
            return '.doc'
        elif 'pdf' in basename_lower:
            return '.pdf'
        elif 'txt' in basename_lower:
            return '.txt'
        elif 'md' in basename_lower:
            return '.md'
        else:
            # Check if there's any extension-like pattern
            if '.' in basename and len(basename.split('.')[-1]) <= 5:
                potential_ext = '.' + basename.split('.')[-1].lower()
                if potential_ext in valid_extensions:
                    return potential_ext
            
            # If we can't determine, use .docx as reasonable default for document processing
            logger.warning(f"Could not determine file type for {filename}, using .docx as default")
            return '.docx'
        
    except Exception as e:
        logger.warning(f"Error extracting extension from {filename}: {e}")
        return '.docx'  # Safe default for document processing

async def async_cleanup_temp_file(temp_path: str) -> None:
    """Async cleanup of temporary file."""
    def _cleanup():
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info(f"🗑️ Cleaned up: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {temp_path}: {str(e)}")
    
    await run_in_threadpool(_cleanup)
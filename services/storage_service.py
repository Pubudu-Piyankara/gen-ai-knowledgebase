import os
import logging
from typing import Optional, Tuple
import tempfile
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase not installed. Only local storage will be available.")

class SupabaseStorage:
    """Handles file storage using Supabase Storage"""
    
    def __init__(self, url: str, key: str, bucket_name: str):
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase client not available")
        
        self.supabase: Client = create_client(url, key)
        self.bucket_name = bucket_name
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            # Try to get bucket info
            buckets = self.supabase.storage.list_buckets()
            bucket_exists = any(bucket.name == self.bucket_name for bucket in buckets)
            
            if not bucket_exists:
                # Create bucket
                self.supabase.storage.create_bucket(self.bucket_name)
                logger.info(f"Created Supabase bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Error checking/creating bucket: {str(e)}")
    
    def upload_file(self, file: FileStorage, filename: str) -> Tuple[bool, str]:
        """
        Upload file to Supabase storage
        Returns: (success, file_path_or_error_message)
        """
        try:
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            # Upload to Supabase
            result = self.supabase.storage.from_(self.bucket_name).upload(
                filename, file_content
            )
            
            if result:
                logger.info(f"Successfully uploaded {filename} to Supabase")
                return True, f"{self.bucket_name}/{filename}"
            else:
                return False, "Upload failed"
                
        except Exception as e:
            logger.error(f"Error uploading file to Supabase: {str(e)}")
            return False, str(e)
    
    def download_file(self, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Download file from Supabase and save to temporary location
        Returns: (success, local_temp_path_or_none)
        """
        try:
            # Download file content
            response = self.supabase.storage.from_(self.bucket_name).download(filename)
            
            if response:
                # Save to temporary file
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, os.path.basename(filename))
                
                with open(temp_path, 'wb') as f:
                    f.write(response)
                
                logger.info(f"Downloaded {filename} to {temp_path}")
                return True, temp_path
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Error downloading file from Supabase: {str(e)}")
            return False, None
    
    def delete_file(self, filename: str) -> bool:
        """Delete file from Supabase storage"""
        try:
            result = self.supabase.storage.from_(self.bucket_name).remove([filename])
            logger.info(f"Deleted {filename} from Supabase")
            return True
        except Exception as e:
            logger.error(f"Error deleting file from Supabase: {str(e)}")
            return False
    
    def get_public_url(self, filename: str) -> Optional[str]:
        """Get public URL for a file"""
        try:
            result = self.supabase.storage.from_(self.bucket_name).get_public_url(filename)
            return result
        except Exception as e:
            logger.error(f"Error getting public URL: {str(e)}")
            return None
    
    def list_files(self) -> list:
        """List all files in the bucket"""
        try:
            files = self.supabase.storage.from_(self.bucket_name).list()
            return files
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []

class SupabaseStorage:
    def __init__(self, url: str, key: str, bucket_name: str):
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase client not available")
        
        self.supabase: Client = create_client(url, key)
        self.bucket_name = bucket_name
    
    def upload_file(self, file: FileStorage, filename: str) -> Tuple[bool, str]:
        try:
            file_content = file.read()
            file.seek(0)
            result = self.supabase.storage.from_(self.bucket_name).upload(filename, file_content)
            if result:
                logger.info(f"Successfully uploaded {filename} to Supabase")
                return True, f"{self.bucket_name}/{filename}"
            else:
                return False, "Upload failed"
        except Exception as e:
            logger.error(f"Error uploading file to Supabase: {str(e)}")
            return False, str(e)
    
    def download_file(self, filename: str) -> Tuple[bool, Optional[str]]:
        try:
            response = self.supabase.storage.from_(self.bucket_name).download(filename)
            if response:
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, os.path.basename(filename))
                with open(temp_path, 'wb') as f:
                    f.write(response)
                logger.info(f"Downloaded {filename} to {temp_path}")
                return True, temp_path
            else:
                return False, None
        except Exception as e:
            logger.error(f"Error downloading file from Supabase: {str(e)}")
            return False, None
    
    def delete_file(self, filename: str) -> bool:
        try:
            result = self.supabase.storage.from_(self.bucket_name).remove([filename])
            logger.info(f"Deleted {filename} from Supabase")
            return True
        except Exception as e:
            logger.error(f"Error deleting file from Supabase: {str(e)}")
            return False
    
    def get_public_url(self, filename: str) -> Optional[str]:
        try:
            result = self.supabase.storage.from_(self.bucket_name).get_public_url(filename)
            return result
        except Exception as e:
            logger.error(f"Error getting public URL: {str(e)}")
            return None
    
    def list_files(self) -> list:
        try:
            files = self.supabase.storage.from_(self.bucket_name).list()
            return files
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []

# StorageManager class
class StorageManager:
    def __init__(self, config):
        self.config = config
        self.use_supabase = getattr(config, 'USE_SUPABASE_STORAGE', False)
        
        if self.use_supabase:
            supabase_url = getattr(config, 'SUPABASE_URL', None)
            supabase_key = getattr(config, 'SUPABASE_KEY', None)
            
            if not supabase_url or not supabase_key:
                logger.warning("Supabase URL or KEY not configured. Falling back to local storage.")
                self.use_supabase = False
            elif not SUPABASE_AVAILABLE:
                logger.warning("Supabase client not available. Falling back to local storage.")
                self.use_supabase = False
        
        if self.use_supabase:
            try:
                self.supabase_storage = SupabaseStorage(
                    url=config.SUPABASE_URL,
                    key=config.SUPABASE_KEY,
                    bucket_name=getattr(config, 'SUPABASE_BUCKET_NAME', 'uploads')
                )
                logger.info("Supabase storage initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase storage: {str(e)}")
                self.use_supabase = False
                logger.info("Falling back to local storage")
        
        upload_folder = getattr(config, 'UPLOAD_FOLDER', 'Uploads')
        os.makedirs(upload_folder, exist_ok=True)
    
    def save_file(self, file: FileStorage, filename: str) -> Tuple[bool, str]:
        if self.use_supabase:
            success, result = self.supabase_storage.upload_file(file, filename)
            if success:
                return True, result
            else:
                logger.warning("Supabase upload failed, falling back to local storage")
        
        try:
            upload_folder = getattr(self.config, 'UPLOAD_FOLDER', 'Uploads')
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            logger.info(f"Saved file locally: {filepath}")
            return True, filepath
        except Exception as e:
            logger.error(f"Local file save failed: {str(e)}")
            return False, str(e)
    
    def get_file_for_processing(self, filename: str) -> Optional[str]:
        if self.use_supabase:
            success, temp_path = self.supabase_storage.download_file(filename)
            if success:
                return temp_path
            else:
                logger.warning("Failed to download from Supabase")
        
        upload_folder = getattr(self.config, 'UPLOAD_FOLDER', 'Uploads')
        local_path = os.path.join(upload_folder, filename)
        if os.path.exists(local_path):
            return local_path
        return None
    
    def cleanup_file(self, filepath: str):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
        except Exception as e:
            logger.error(f"Error cleaning up file {filepath}: {str(e)}")
    
    def store_original_file(self, file, document_id: str):
        """Store the original file for future reference"""
        try:
            # Create a unique filename with document_id
            original_filename = getattr(file, 'filename', 'unknown_file')
            filename = f"{document_id}_{original_filename}"
            
            # Reset file pointer to beginning
            file.seek(0)
            
            success, result = self.save_file(file, filename)
            if success:
                logger.info(f"Stored original file: {filename}")
            else:
                logger.error(f"Failed to store original file: {result}")
        except Exception as e:
            logger.error(f"Error storing original file: {str(e)}")
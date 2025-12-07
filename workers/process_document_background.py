import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from services.async_document_processor import DocumentProcessor
from services.file_service import async_cleanup_temp_file
from services.memory_service import async_update_memory_status
from services.async_vector_store import ZillizVectorStoreNew
from config import Config
# from services.vector_store import /

logger = logging.getLogger(__name__)

async def process_document_background(
    temp_path: str,
    memory_id: str,
    space_id: str,
    file_name: str,
    file_type: str,
    file_size: int,
    storage_method: str,
    file_url: str
) -> None:
    """
    Background task to process document: parse, chunk, embed, and store in vector DB.
    """
    
    try:
        logger.info(f"Starting background processing for memory {memory_id}")
        
        # Update status to processing
        await async_update_memory_status(memory_id, "processing")
        
        # Initialize config and services
        config = Config()
        
        logger.info(f"Processing document: {file_name}")
        
        # Create document processor
        document_processor = DocumentProcessor()
        
        # Use the original filename for content processing, not the temp path
        # The processor should use the original filename to determine file type
        logger.info(f"📄 Processing file: {file_name} (temp path: {temp_path})")
        
        # Process the document - pass the original filename for type detection
        processed_content = document_processor.process_file(temp_path, original_filename=file_name)
        
        if not processed_content:
            logger.error(f"No content extracted from {file_name}")
            raise ValueError("Failed to process document")
        
        logger.info(f"✅ Extracted {len(processed_content)} characters from document")
        
        # Initialize vector store - get singleton instance and ensure initialization
        vector_store = ZillizVectorStoreNew(config)
        await vector_store.initialize()  # Explicit initialization
        
        # Create chunks and embeddings
        logger.info("📦 Creating chunks and embeddings...")
        chunks = document_processor.chunk_content(
            content=processed_content,
            file_name=file_name,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Store in vector database
        logger.info("💾 Storing in vector database...")
        success = await vector_store.async_add_documents(
            chunks=chunks,
            space_id=space_id,
            memory_id=memory_id,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            storage_method=storage_method,
            file_url=file_url
        )
        
        if success:
            logger.info(f"✅ Successfully processed and stored document for memory {memory_id}")
            await async_update_memory_status(memory_id, "completed")
        else:
            logger.error(f"❌ Failed to store document in vector database for memory {memory_id}")
            await async_update_memory_status(memory_id, "failed")
            
    except Exception as e:
        logger.error(f"❌ Background processing failed for memory {memory_id}: {str(e)}", exc_info=True)
        await async_update_memory_status(memory_id, "failed")
    
    finally:
        # Always cleanup the temporary file
        if temp_path:
            # Resolve the real path to prevent path traversal attacks
            real_path = os.path.realpath(temp_path)
            # Verify the resolved path is within allowed temp directories
            allowed_dirs = ['/tmp/', '/var/folders/']
            is_valid = any(real_path.startswith(d) or real_path == d.rstrip('/') for d in allowed_dirs)
            
            if is_valid:
                await async_cleanup_temp_file(real_path)
            else:
                logger.warning(f"Suspicious temp path, not cleaning up: {temp_path} (resolved: {real_path})")
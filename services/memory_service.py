# Helper function to update memory status
from fastapi import logger
from datetime import datetime
from fastapi.concurrency import run_in_threadpool
from supabase import create_client, Client
from config import Config

config = Config()

import logging
logger = logging.getLogger(__name__)

async def async_update_memory_status(memory_id: str, status: str) -> bool:
    """Async wrapper for updating memory status in Supabase."""
    def _update_status():
        try:
            supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            update_response = supabase.table("memories").update({
                "processing_status": status,
                "updated_at": datetime.now().isoformat()
            }).eq("id", memory_id).execute()
            
            logger.info(f"Updated memory {memory_id} status to '{status}'")
            return True
        except Exception as e:
            logger.error(f"Error updating memory {memory_id} status: {str(e)}")
            return False
    
    return await run_in_threadpool(_update_status)

def update_memory_status(supabase, memory_id: str, status: str) -> bool:
    """Update memory processing status in Supabase."""
    try:
        # Validate inputs
        if not supabase:
            logger.warning(f"Supabase client is None, cannot update memory {memory_id}")
            return False
            
        if not memory_id:
            logger.warning("Memory ID is empty, cannot update status")
            return False
        
        # Log the update attempt
        logger.info(f"Attempting to update memory {memory_id} status to '{status}'")
        
        # Perform the update with error handling
        update_response = supabase.table("memories").update({
            "processing_status": status,
            "updated_at": datetime.now().isoformat()
        }).eq("id", memory_id).execute()
        
        # Check if update was successful - Supabase returns empty data for updates sometimes
        if hasattr(update_response, 'data') and update_response.data is not None:
            if len(update_response.data) > 0:
                logger.info(f"Successfully updated memory {memory_id} status to '{status}' - Data: {update_response.data}")
                return True
            else:
                # Empty data doesn't necessarily mean failure in Supabase
                logger.info(f"Update completed for memory {memory_id} status to '{status}' (empty response is normal)")
                return True
        else:
            logger.warning(f"Unexpected response format when updating memory {memory_id}")
            return False
        
    except Exception as e:
        # Log the full error with traceback
        logger.error(f"Error updating memory {memory_id} status to '{status}': {str(e)}", exc_info=True)
        return False
 

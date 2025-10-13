import base64
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from httpcore import request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import tempfile
import os
import shutil
import logging

from datetime import datetime
from pathlib import Path

import requests

from services.document_processor import DocumentProcessor
from services.vector_store import ZillizVectorStore
from services.query_service import QueryService
from services.storage_service import StorageManager
from config import Config
from supabase import create_client, Client
from werkzeug.utils import secure_filename
from fastapi import Request
from services.memory_retrieval import (
    retrieve_file_content,
    get_file_mime_type,
    validate_file_size,
    prepare_multipart_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG System API",
    description="A comprehensive RAG system with document processing, vectorization, and intelligent querying",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Initialize services
config = Config()
document_processor = DocumentProcessor()
vector_store = ZillizVectorStore(
    host=config.MILVUS_HOST,
    port=config.MILVUS_PORT,
    uri=config.MILVUS_URI,
    token=config.MILVUS_TOKEN,
    collection_name=config.COLLECTION_NAME
)
query_service = QueryService(vector_store, GEMINI_KEY=config.GEMINI_KEY)
storage_manager = StorageManager(config)

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    filter_expr: Optional[str] = Field(default=None, description="Milvus filter expression")
    include_metadata: bool = Field(default=True, description="Include metadata in results")

class CreateMemoryRequest(BaseModel):
    space_id: str = Field(..., description="Space UUID")
    user_id: str = Field(..., description="User UUID")
    title: str = Field(..., description="Memory title")
    type: str = Field(..., description="Memory type: upload, url, youtube, website, audio_recording, text")
    content: Optional[str] = Field(default=None, description="Text content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    file_url: Optional[str] = Field(default=None, description="File URL for uploads")

class ChatWithMemoryRequest(BaseModel):
    memory_id: str = Field(..., description="Memory UUID to chat with")
    user_id: str = Field(..., description="User UUID")
    message: str = Field(..., description="User's question or message")

class GenerateFlashcardsRequest(BaseModel):
    memory_id: str = Field(..., description="Memory UUID to generate flashcards from")
    user_id: str = Field(..., description="User UUID")
    count: int = Field(default=5, ge=1, le=20, description="Number of flashcards to generate")

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    response: Optional[str] = None
    total_results: int
    processing_time: float

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    metadata: Dict[str, Any]
    chunks_count: int
    upload_time: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        services={
            "document_processor": "active",
            "vector_store": "active",
            "storage": "active"
        }
    )
    
    
def validate_file_parameters(type: str, storageMethod: str, fileName: str, fileUrl: str) -> None:
    """Validate incoming file parameters."""
    if type != "upload":
        raise HTTPException(status_code=400, detail="Invalid type, must be 'upload'")
    
    if storageMethod not in ["storage", "database"]:
        raise HTTPException(status_code=400, detail="Invalid storage method")
    
    if not fileName:
        raise HTTPException(status_code=400, detail="fileName is required")
    
    if not fileUrl:
        raise HTTPException(status_code=400, detail="fileUrl is required")

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

# Fix the validate_request_data function for FastAPI
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
    
# Helper function to update memory status
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
    
# Helper function to cleanup temporary file
def cleanup_temp_file(temp_path: str) -> None:
    """Safely cleanup temporary file."""
    try:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info(f"Cleaned up temporary file: {temp_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {temp_path}: {str(e)}")

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
    
@app.post("/api/process-memory")
async def process_memory(
    file: UploadFile = File(...),
    type: str = Form(...),
    fileName: str = Form(...),
    fileType: str = Form(...),
    fileSize: str = Form(...),
    spaceId: str = Form(...),
    storageMethod: str = Form(...),
    memoryId: str = Form(...)
):
    """
    Process a file uploaded directly from Next.js into the RAG system.
    """
    
    temp_path = None
    supabase = None
    
    try:
        # Log incoming request
        logger.info(f"Received file upload request")
        logger.info(f"File name: {fileName}")
        logger.info(f"File type: {fileType}")
        logger.info(f"Memory ID: {memoryId}")
        
        # Create form_data and files dict for validation
        form_data = {
            'type': type,
            'fileName': fileName,
            'fileType': fileType,
            'spaceId': spaceId,
            'storageMethod': storageMethod,
            'memoryId': memoryId
        }
        files = {'file': file}
        
        # Validate request data
        await validate_request_data(form_data, files)
        
        logger.info(f"Processing file: {fileName}")
        logger.info(f"File type: {fileType}")
        logger.info(f"Reported size: {fileSize}")
        logger.info(f"Memory ID: {memoryId}")
        logger.info(f"Storage method: {storageMethod}")
        
        # Initialize Supabase client
        supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        
        # Update status to "processing"
        update_memory_status(supabase, memoryId, "processing")
        
        # Save uploaded file to temporary location
        temp_path, actual_file_size = await save_uploaded_file(file, fileName)
        
        logger.info(f"File saved successfully. Actual size: {actual_file_size} bytes")
        
        # Validate file
        validate_file(temp_path, fileName)
        
        # Process document
        logger.info(f"Starting document processing for {fileName}")
        
        processed_data = document_processor.process_document(temp_path)
        
        if not processed_data:
            raise ValueError("Failed to process document")
        
        logger.info(f"Document processed successfully - Chunks created: {len(processed_data['content'])}")
        
        # Enhance metadata with additional information
        metadata = processed_data.get('metadata', {})
        metadata.update({
            'space_id': spaceId,
            'memory_id': memoryId,
            'file_type': fileType,
            'original_name': Path(fileName).name,
            'storage_method': storageMethod,
            'file_size': actual_file_size,
            'chunks_count': len(processed_data['content']),
            'processed_via': 'direct_upload'
        })
        
        # Store in vector store
        logger.info(f"Storing document in vector store: {fileName}")
        document_id = vector_store.store_document(
            filename=fileName,
            chunks=processed_data['content'],
            metadata=metadata
        )
        
        logger.info(f"Document stored successfully - Document ID: {document_id}")
        
        # Update Supabase status to "completed"
        update_memory_status(supabase, memoryId, "completed")
        
        # Prepare success response
        response_data = {
            "message": "File processed and added to RAG system successfully",
            "document_id": document_id,
            "space_id": spaceId,
            "memory_id": memoryId,
            "chunks_created": len(processed_data['content']),
            "file_size": actual_file_size,
            "storage_method": storageMethod,
            "processing_method": "direct_upload"
        }
        
        logger.info(f"Processing completed successfully for memory {memoryId}")
        return JSONResponse(content=response_data, status_code=200)
        
    except ValueError as ve:
        # Handle validation and processing errors
        logger.error(f"Validation/Processing error: {str(ve)}")
        cleanup_temp_file(temp_path)
        if memoryId and supabase:
            update_memory_status(supabase, memoryId, "failed")
        return JSONResponse(content={"error": str(ve)}, status_code=400)
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error processing memory: {str(e)}", exc_info=True)
        cleanup_temp_file(temp_path)
        if memoryId and supabase:
            update_memory_status(supabase, memoryId, "failed")
        
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=500)
    
    finally:
        # Always cleanup temporary file
        cleanup_temp_file(temp_path)
        
               
@app.post("/api/ingest-upload")
async def ingest_upload(
    type: str = Form(...),
    memoryId: str = Form(...),
    fileUrl: str = Form(...),
    storageMethod: str = Form(...),
    spaceId: str = Form(...),
    fileName: str = Form(...),
    fileType: str = Form(...),
    fileSize: str = Form(...)
):
    """
    Ingest a file from Supabase storage or base64 data and forward to processing endpoint.
    """
    
    supabase = None
    temp_path = None
    
    try:
        # Validate required fields
        if not all([memoryId, fileUrl, storageMethod, spaceId, fileName]):
            raise HTTPException(
                status_code=400, 
                detail="Missing required fields: memoryId, fileUrl, storageMethod, spaceId, fileName"
            )
        
        # Validate storage method
        if storageMethod not in ['storage', 'database']:
            raise HTTPException(
                status_code=400,
                detail="Invalid storageMethod. Must be 'storage' or 'database'"
            )
        
        logger.info(f"Starting file ingestion for memory {memoryId}")
        logger.info(f"Storage method: {storageMethod}, File: {fileName}")
        logger.info(f"File URL: {fileUrl}")
        
        correct_supabase_url = config.SUPABASE_URL
        supabase_key = config.SUPABASE_KEY
        bucket_name = config.SUPABASE_BUCKET_NAME or "uploads"
        
        logger.info(f"Using Supabase URL: {correct_supabase_url}")
        
        # Initialize Supabase client with correct URL
        try:
            supabase = create_client(correct_supabase_url, supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}", exc_info=True)
            logger.warning("Proceeding without database status updates")
            supabase = None
        
        # Update memory status to 'processing'
        if supabase:
            status_updated = update_memory_status(supabase, memoryId, "processing")
            if not status_updated:
                logger.warning(f"Failed to update memory status, but continuing...")
        
        # Retrieve file content
        file_content = None
        try:
            logger.info(f"Attempting to retrieve file from: {fileUrl}")
            
            # if storageMethod == 'storage':
            #     # Extract bucket and path from URL
            #     # URL format: https://PROJECT.supabase.co/storage/v1/object/public/BUCKET/PATH
            #     url_parts = fileUrl.split('/storage/v1/object/public/')
                
            #     if len(url_parts) == 2:
            #         # Parse bucket and file path
            #         path_parts = url_parts[1].split('/', 1)
            #         bucket_name = path_parts[0]
            #         file_path = path_parts[1] if len(path_parts) > 1 else ''
                    
            #         logger.info(f"Parsed URL - Bucket: {bucket_name}, Path: {file_path}")
                    
            #         # Try multiple approaches to download the file
            #         download_success = False
                    
            #         # Approach 1: Try Supabase API first
            #         if supabase and not download_success:
            #             try:
            #                 logger.info("Attempting download via Supabase storage API...")
            #                 file_content = supabase.storage.from_(bucket_name).download(file_path)
            #                 if file_content and len(file_content) > 0:
            #                     logger.info(f"✅ Successfully downloaded via Supabase API, size: {len(file_content)} bytes")
            #                     download_success = True
            #                 else:
            #                     logger.warning("Supabase API returned empty content")
            #             except Exception as api_error:
            #                 logger.warning(f"Supabase API failed: {str(api_error)}")
                    
            #         # Approach 2: Try direct HTTP download from public URL
            #         if not download_success:
            #             try:
            #                 logger.info("Attempting direct HTTP download from public URL...")
            #                 import requests
            #                 response = requests.get(fileUrl)
            #                 response.raise_for_status()
            #                 file_content = response.content
            #                 if len(file_content) > 0:
            #                     logger.info(f"✅ Successfully downloaded via HTTP, size: {len(file_content)} bytes")
            #                     download_success = True
            #                 else:
            #                     logger.warning("HTTP download returned empty content")
            #             except requests.exceptions.RequestException as http_error:
            #                 logger.warning(f"HTTP download failed: {str(http_error)}")
                    
            #         # Approach 3: Try authenticated download
            #         if not download_success and supabase:
            #             try:
            #                 logger.info("Attempting authenticated download...")
            #                 headers = {
            #                     'Authorization': f'Bearer {supabase_key}',
            #                     'apikey': supabase_key
            #                 }
            #                 response = requests.get(fileUrl, headers=headers, timeout=30)
            #                 response.raise_for_status()
            #                 file_content = response.content
            #                 if len(file_content) > 0:
            #                     logger.info(f"✅ Successfully downloaded with auth, size: {len(file_content)} bytes")
            #                     download_success = True
            #             except Exception as auth_error:
            #                 logger.warning(f"Authenticated download failed: {str(auth_error)}")
                    
            #         if not download_success:
            #             raise ValueError(f"All download methods failed. File may not exist at: {fileUrl}")
                    
            #     else:
            #         # Malformed URL, try direct download
            #         logger.info("URL format unexpected, trying direct download...")
            #         response = requests.get(fileUrl, timeout=30)
            #         response.raise_for_status()
            #         file_content = response.content
            #         logger.info(f"Downloaded file, size: {len(file_content)} bytes")
            if storageMethod == 'storage':
                download_success = False
                
                # METHOD 1: Direct HTTP download (works with signed URLs and public URLs)
                try:
                    logger.info("Method 1: Attempting direct HTTP download...")
                    response = requests.get(fileUrl, timeout=30)
                    response.raise_for_status()
                    file_content = response.content
                    if len(file_content) > 0:
                        logger.info(f"✅ Successfully downloaded via HTTP, size: {len(file_content)} bytes")
                        download_success = True
                except requests.exceptions.RequestException as http_error:
                    logger.warning(f"HTTP download failed: {str(http_error)}")
                    
            else:
                # Database storage method (base64)
                file_content = retrieve_file_content(fileUrl, storageMethod)
                logger.info(f"Retrieved file content from database, size: {len(file_content)} bytes")
                
        except Exception as e:
            logger.error(f"Error retrieving file: {str(e)}", exc_info=True)
            if supabase:
                update_memory_status(supabase, memoryId, "failed")
            
            # Provide more specific error messages
            if "not_found" in str(e) or "404" in str(e):
                error_msg = f"File not found at the specified location. The file may have been deleted or the URL is incorrect. URL: {fileUrl}"
            elif "403" in str(e) or "Forbidden" in str(e):
                error_msg = f"Access denied. The file may be private or the authentication is incorrect."
            else:
                error_msg = f"Failed to download file: {str(e)}"
            
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        if not file_content or len(file_content) == 0:
            if supabase:
                update_memory_status(supabase, memoryId, "failed")
            raise HTTPException(status_code=500, detail="File content is empty")
        
        # Validate file size
        try:
            validate_file_size(file_content, config.MAX_CONTENT_LENGTH)
        except ValueError as e:
            if supabase:
                update_memory_status(supabase, memoryId, "failed")
            raise HTTPException(status_code=413, detail=str(e))
        
        # Save to temporary file for processing
        try:
            import tempfile
            temp_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(fileName).suffix,
                prefix="ingest_"
            ).name
            
            with open(temp_path, "wb") as f:
                f.write(file_content)
            
            logger.info(f"Saved file to temporary path: {temp_path}")
            
        except Exception as e:
            logger.error(f"Failed to save temporary file: {str(e)}")
            if supabase:
                update_memory_status(supabase, memoryId, "failed")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file: {str(e)}"
            )
        
        # Process document directly
        try:
            logger.info(f"Starting document processing for {fileName}")
            
            processed_data = document_processor.process_document(temp_path)
            
            if not processed_data:
                raise ValueError("Failed to process document")
            
            logger.info(f"Document processed successfully - Chunks created: {len(processed_data['content'])}")
            
            # Enhance metadata
            metadata = processed_data.get('metadata', {})
            metadata.update({
                'space_id': spaceId,
                'memory_id': memoryId,
                'file_type': fileType,
                'original_name': Path(fileName).name,
                'storage_method': storageMethod,
                'file_size': len(file_content),
                'chunks_count': len(processed_data['content']),
                'processed_via': 'ingestion',
                'file_url': fileUrl
            })
            
            # Store in vector store
            logger.info(f"Storing document in vector store: {fileName}")
            document_id = vector_store.store_document(
                filename=fileName,
                chunks=processed_data['content'],
                metadata=metadata
            )
            
            logger.info(f"Document stored successfully - Document ID: {document_id}")
            
            # Update memory status to 'completed'
            if supabase:
                update_memory_status(supabase, memoryId, "completed")
            
            return JSONResponse({
                "message": "File ingested and processed successfully",
                "memory_id": memoryId,
                "document_id": document_id,
                "chunks_created": len(processed_data['content']),
                "file_size": len(file_content),
                "storage_method": storageMethod,
                "processing_method": "ingestion"
            }, status_code=200)
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            if supabase:
                update_memory_status(supabase, memoryId, "failed")
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {str(e)}"
            )
    
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in file ingestion: {str(e)}", exc_info=True)
        
        if supabase and memoryId:
            update_memory_status(supabase, memoryId, "failed")
        
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )
    
    finally:
        cleanup_temp_file(temp_path)
        
                
@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process documents"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not document_processor.is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed types: {document_processor.allowed_extensions}"
        )
    
    # Check file size
    if file.size > config.MAX_CONTENT_LENGTH:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Process document
        processed_data = document_processor.process_document(tmp_path)
        
        if not processed_data:
            os.unlink(tmp_path)
            raise HTTPException(status_code=500, detail="Failed to process document")
        
        # Store processed document
        document_id = vector_store.store_document(
            filename=file.filename,
            chunks=processed_data['content'],
            metadata=processed_data['metadata']
        )
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Optionally store original file in background
        if config.STORE_ORIGINAL_FILES:
            background_tasks.add_task(
                storage_manager.store_original_file, 
                file, 
                document_id
            )
        
        return JSONResponse({
            "message": "Document uploaded and processed successfully",
            "document_id": document_id,
            "filename": file.filename,
            "chunks_created": len(processed_data['content']),
            "document_type": processed_data.get('document_type'),
            "metadata": processed_data['metadata']
        })
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the indexed documents"""
    
    start_time = datetime.now()
    
    try:
        # Perform similarity search
        results = vector_store.similarity_search(
            query=request.query,
            top_k=request.top_k,
            filter_expr=request.filter_expr
        )
        
        logger.info(f"Similarity search returned {len(results)} results")
        if results:
            logger.info(f"First result keys: {list(results[0].keys()) if results else 'No results'}")
        
        # Generate AI response if configured
        ai_response = None
        if config.GEMINI_KEY and results:
            logger.info("Attempting to generate AI response with Gemini")
            ai_response = query_service.generate_response(request.query, results)
            logger.info(f"Generated response: {ai_response[:100]}..." if ai_response else "No response generated")
        else:
            logger.warning(f"AI response not generated. Gemini key available: {bool(config.GEMINI_KEY)}, Results: {len(results)}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            query=request.query,
            results=results,
            response=ai_response,
            total_results=len(results),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Query processing failed")

@app.get("/api/documents")
async def list_documents():
    """List all indexed documents"""
    try:
        # This would require additional tracking in your vector store
        # For now, return collection stats
        stats = vector_store.get_collection_stats()
        return JSONResponse(stats)
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document"""
    try:
        success = vector_store.delete_document(document_id)
        if success:
            return JSONResponse({"message": "Document deleted successfully"})
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        stats = vector_store.get_collection_stats()
        return JSONResponse(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

# Memory Management Endpoints

@app.post("/api/memories")
async def create_memory(request: CreateMemoryRequest):
    """Create a new memory entry in the database"""
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    try:
        response = supabase.table("memories").insert({
            "space_id": request.space_id,
            "user_id": request.user_id,
            "title": request.title,
            "type": request.type,
            "content": request.content,
            "metadata": request.metadata or {},
            "file_url": request.file_url,
            "processing_status": "pending" if request.type == "upload" else "completed"
        }).execute()
        
        if response.data:
            return JSONResponse({
                "message": "Memory created successfully",
                "memory": response.data[0]
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to create memory")
            
    except Exception as e:
        logger.error(f"Error creating memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {str(e)}")

@app.get("/api/memories/{memory_id}")
async def get_memory(memory_id: str, user_id: str):
    """Get a specific memory by ID"""
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    try:
        response = supabase.table("memories").select("*").eq("id", memory_id).eq("user_id", user_id).execute()
        
        if response.data:
            return JSONResponse(response.data[0])
        else:
            raise HTTPException(status_code=404, detail="Memory not found")
            
    except Exception as e:
        logger.error(f"Error fetching memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch memory: {str(e)}")

@app.get("/api/spaces/{space_id}/memories")
async def get_space_memories(space_id: str, user_id: str):
    """Get all memories for a specific space"""
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    try:
        response = supabase.table("memories").select("*").eq("space_id", space_id).eq("user_id", user_id).order("created_at", desc=True).execute()
        
        return JSONResponse({
            "memories": response.data,
            "total": len(response.data)
        })
            
    except Exception as e:
        logger.error(f"Error fetching space memories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch memories: {str(e)}")

@app.post("/api/memories/{memory_id}/chat")
async def chat_with_memory(memory_id: str, request: ChatWithMemoryRequest):
    """Chat with a specific memory using RAG"""
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    try:
        # Get memory metadata
        memory_response = supabase.table("memories").select("*").eq("id", memory_id).eq("user_id", request.user_id).execute()
        
        if not memory_response.data:
            raise HTTPException(status_code=404, detail="Memory not found")
            
        memory = memory_response.data[0]
        
        # Search for relevant content in vector store
        search_results = vector_store.similarity_search(
            query=request.message,
            top_k=5,
            filter_expr=f'memory_id == "{memory_id}"'
        )
        
        # Generate AI response if OpenAI key is available
        ai_response = None
        if config.GEMINI_KEY and search_results:
            ai_response = query_service.generate_response(request.message, search_results)
        else:
            # Fallback response
            ai_response = f"I found {len(search_results)} relevant sections in the document. Here's what I found: " + \
                         " ".join([result.get('content', '')[:200] + "..." for result in search_results[:2]])
        
        # Store chat history
        chat_response = supabase.table("memory_chats").insert({
            "memory_id": memory_id,
            "user_id": request.user_id,
            "message": request.message,
            "response": ai_response
        }).execute()
        
        return JSONResponse({
            "message": request.message,
            "response": ai_response,
            "memory_title": memory["title"],
            "relevant_chunks": len(search_results)
        })
        
    except Exception as e:
        logger.error(f"Error in memory chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/api/memories/{memory_id}/flashcards")
async def generate_flashcards(memory_id: str, request: GenerateFlashcardsRequest):
    """Generate flashcards from memory content"""
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    try:
        # Get memory content
        memory_response = supabase.table("memories").select("*").eq("id", memory_id).eq("user_id", request.user_id).execute()
        
        if not memory_response.data:
            raise HTTPException(status_code=404, detail="Memory not found")
            
        memory = memory_response.data[0]
        
        # Get content from vector store
        search_results = vector_store.similarity_search(
            query="summary overview main points",  # Query to get comprehensive content
            top_k=20,
            filter_expr=f'memory_id == "{memory_id}"'
        )
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No content found for this memory")
        
        # Combine content for flashcard generation
        combined_content = " ".join([result.get('content', '') for result in search_results])
        
        # Generate flashcards using AI (if available)
        flashcards = []
        if config.GEMINI_KEY:
            # Use AI to generate flashcards
            prompt = f"""
            Based on the following content, generate {request.count} educational flashcards.
            Each flashcard should have a clear question and a concise answer.
            Focus on key concepts, definitions, and important facts.
            
            Content: {combined_content[:3000]}...
            
            Return the flashcards in this format:
            Q: [Question]
            A: [Answer]
            """
            
            ai_response = query_service.generate_response(prompt, [])
            
            # Parse AI response to extract flashcards
            lines = ai_response.split('\n')
            current_question = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Q:'):
                    current_question = line[2:].strip()
                elif line.startswith('A:') and current_question:
                    answer = line[2:].strip()
                    flashcards.append({
                        "question": current_question,
                        "answer": answer,
                        "difficulty_level": "medium"
                    })
                    current_question = None
        else:
            # Fallback: Create simple flashcards from key sentences
            sentences = combined_content.split('.')[:request.count]
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) > 20:
                    flashcards.append({
                        "question": f"What is discussed regarding: {sentence.strip()[:50]}...?",
                        "answer": sentence.strip(),
                        "difficulty_level": "medium"
                    })
        
        # Store flashcards in database
        stored_flashcards = []
        for flashcard in flashcards[:request.count]:
            response = supabase.table("flashcards").insert({
                "memory_id": memory_id,
                "user_id": request.user_id,
                "question": flashcard["question"],
                "answer": flashcard["answer"],
                "difficulty_level": flashcard["difficulty_level"]
            }).execute()
            
            if response.data:
                stored_flashcards.append(response.data[0])
        
        return JSONResponse({
            "message": f"Generated {len(stored_flashcards)} flashcards",
            "flashcards": stored_flashcards,
            "memory_title": memory["title"]
        })
        
    except Exception as e:
        logger.error(f"Error generating flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {str(e)}")

@app.get("/api/memories/{memory_id}/analytics")
async def get_memory_analytics(memory_id: str, user_id: str):
    """Get analytics for a specific memory"""
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    try:
        # Get memory details
        memory_response = supabase.table("memories").select("*").eq("id", memory_id).eq("user_id", user_id).execute()
        
        if not memory_response.data:
            raise HTTPException(status_code=404, detail="Memory not found")
            
        memory = memory_response.data[0]
        
        # Get flashcard count
        flashcard_response = supabase.table("flashcards").select("*", count="exact").eq("memory_id", memory_id).execute()
        flashcard_count = flashcard_response.count or 0
        
        # Get chat count
        chat_response = supabase.table("memory_chats").select("*", count="exact").eq("memory_id", memory_id).execute()
        chat_count = chat_response.count or 0
        
        # Get vector store stats
        vector_stats = vector_store.get_collection_stats()
        
        return JSONResponse({
            "memory": {
                "id": memory["id"],
                "title": memory["title"],
                "type": memory["type"],
                "processing_status": memory["processing_status"],
                "created_at": memory["created_at"]
            },
            "statistics": {
                "flashcards_generated": flashcard_count,
                "chat_interactions": chat_count,
                "vector_chunks": vector_stats.get("num_entities", 0),
                "last_interaction": memory["updated_at"]
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting memory analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
    
def retrieve_file_from_supabase(
    supabase_url: str,
    supabase_key: str,
    bucket_name: str,
    file_path: str,
    destination_path: Optional[str] = None
) -> Tuple[bytes, str]:
    """
    Retrieve a file from Supabase Storage.
    """
    try:
        # Initialize Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)

        # Get file name from path
        file_name = os.path.basename(file_path)

        # Download file from Supabase Storage
        response = supabase.storage.from_(bucket_name).download(file_path)

        if response is None:
            logger.error(f"File not found: {file_path} in bucket {bucket_name}")
            raise HTTPException(status_code=404, detail=f"File {file_path} not found in bucket {bucket_name}")

        # If destination_path is provided, save the file locally
        if destination_path:
            try:
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                with open(destination_path, 'wb') as f:
                    f.write(response)
                logger.info(f"File saved to {destination_path}")
            except Exception as e:
                logger.error(f"Failed to save file to {destination_path}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        logger.info(f"Successfully retrieved file: {file_path} from bucket {bucket_name}")
        return response, file_name

    except Exception as e:
        logger.error(f"Error retrieving file from Supabase: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {str(e)}")



    
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, BackgroundTasks, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import tempfile
import os
import shutil
import logging

from datetime import datetime
from pathlib import Path
import uvicorn

from services.document_processor import DocumentProcessor
from services.file_service import async_download_file, async_save_temp_file, cleanup_temp_file, save_uploaded_file
from services.memory_service import async_update_memory_status, update_memory_status
from services.validation_service import validate_file, validate_request_data,validate_file_size
from services.async_vector_store import ZillizVectorStoreNew
from services.query_service import QueryService
from services.storage_service import StorageManager
from config import Config
from supabase import create_client, Client
from fastapi import Request

from services.vector_store import ZillizVectorStore
from workers.process_document_background import process_document_background
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global HTTP client and vector store
http_client: Optional[httpx.AsyncClient] = None
vector_store: Optional[ZillizVectorStoreNew] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI application.
    Manages startup and shutdown of global resources.
    """
    # Startup
    global http_client, vector_store
    
    try:
        logger.info("🚀 Starting application initialization...")
        
        # Initialize HTTP client
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        logger.info("✅ HTTP client initialized")
        
        # Initialize vector store connection
        try:
            config = Config()
            vector_store = ZillizVectorStoreNew(config)
            logger.info("✅ Vector store instance created (lazy initialization)")
        except Exception as e:
            logger.warning(f"⚠️ Vector store creation failed: {e}")
            vector_store = None
        
        logger.info("✅ Application startup completed successfully")
        
        # Application is now running
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("🔄 Starting application shutdown...")
        
        if http_client:
            try:
                await http_client.aclose()
                logger.info("✅ HTTP client closed")
            except Exception as e:
                logger.error(f"❌ Error closing HTTP client: {e}")
        
        logger.info("✅ Application shutdown completed")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Advanced RAG System API",
    description="A comprehensive RAG system with document processing, vectorization, and intelligent querying",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
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
query_service = QueryService(vector_store=None, GEMINI_KEY=config.GEMINI_KEY)
storage_manager = StorageManager(config)
vector_store_old = ZillizVectorStore(
    host=config.MILVUS_HOST,
    port=config.MILVUS_PORT,
    uri=config.MILVUS_URI,
    token=config.MILVUS_TOKEN,
    collection_name=config.COLLECTION_NAME
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query")
    space_id: str = Field(..., description="Space ID to search within")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    # score_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    filter_expr: Optional[str] = Field(default=None, description="Milvus filter expression")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    
class QueryRequestOld(BaseModel):
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

# Helper function to ensure vector store is available
async def get_vector_store() -> ZillizVectorStoreNew:
    """Get the vector store instance, ensuring it's initialized."""
    global vector_store
    
    if vector_store is None:
        logger.error("Vector store not initialized during startup")
        raise HTTPException(status_code=503, detail="Vector store service not available - initialization failed")
    
    # Ensure the vector store is initialized (lazy initialization)
    try:
        await vector_store._ensure_initialized()
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Vector store initialization failed: {str(e)}")

        
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
    background_tasks: BackgroundTasks,
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
    start_time = datetime.now()
    
    supabase = None
    temp_path = None
    
    try:
        # Step 1: Validate required fields
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
        
        logger.info(f"🚀 Starting async ingestion for memory {memoryId}")
        logger.info(f"File: {fileName}, Storage: {storageMethod}")
        
        # Step 2: Update status to queued
        await async_update_memory_status(memoryId, "queued")
        
        # Step 3: Download file asynchronously
        logger.info(f"📥 Downloading file from: {fileUrl}")
        
        bucket_name = config.SUPABASE_BUCKET_NAME or "uploads"
        file_content = await async_download_file(fileUrl, storageMethod, bucket_name)
        
        if not file_content or len(file_content) == 0:
            await async_update_memory_status(memoryId, "failed")
            raise HTTPException(status_code=400, detail="File content is empty")
        
        # Step 4: Validate file size
        try:
            validate_file_size(file_content, config.MAX_CONTENT_LENGTH)
        except ValueError as e:
            await async_update_memory_status(memoryId, "failed")
            raise HTTPException(status_code=413, detail=str(e))
        
        # Step 5: Save to temporary file asynchronously
        temp_path = await async_save_temp_file(file_content, fileName)
        
        # Step 6: Queue background processing task
        logger.info(f"📋 Queueing background processing for memory {memoryId}")
        
        background_tasks.add_task(
            process_document_background,
            temp_path=temp_path,
            memory_id=memoryId,
            space_id=spaceId,
            file_name=fileName,
            file_type=fileType,
            file_size=len(file_content),
            storage_method=storageMethod,
            file_url=fileUrl
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"⚡ Async ingestion setup completed in {processing_time:.2f}s")
        
        # Step 7: Return immediate response
        return JSONResponse({
            "message": "File download completed, processing queued",
            "memory_id": memoryId,
            "status": "queued",
            "file_size": len(file_content),
            "storage_method": storageMethod,
            "processing_method": "async_ingestion",
            "estimated_processing_time": "30-60 seconds"
        }, status_code=202)  # 202 Accepted
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in async ingestion: {str(e)}", exc_info=True)
        
        if memoryId:
            await async_update_memory_status(memoryId, "failed")
        
        raise HTTPException(
            status_code=500,
            detail=f"Async ingestion failed: {str(e)}"
        )
        
                
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

@app.get("/api/debug/vector-store")
async def debug_vector_store():
    """Debug endpoint to check vector store status and data"""
    global vector_store
    
    try:
        vector_store_instance = await get_vector_store()
        
        # Get collection stats
        def _get_collection_stats():
            try:
                # Get basic collection info
                collection_info = {
                    "collection_name": vector_store_instance.collection_name,
                    "collection_exists": vector_store_instance.collection is not None
                }
                
                if vector_store_instance.collection:
                    # Get entity count
                    collection_info["entity_count"] = vector_store_instance.collection.num_entities
                    
                    # Try to get some sample data
                    try:
                        # Query a few random entities
                        results = vector_store_instance.collection.query(
                            expr="",
                            limit=5,
                            output_fields=["id", "space_id", "memory_id", "file_name", "content"]
                        )
                        
                        collection_info["sample_entities"] = []
                        for result in results:
                            collection_info["sample_entities"].append({
                                "id": result.get("id", ""),
                                "space_id": result.get("space_id", ""),
                                "memory_id": result.get("memory_id", ""),
                                "file_name": result.get("file_name", ""),
                                "content_preview": str(result.get("content", ""))[:100] + "..."
                            })
                    except Exception as e:
                        collection_info["sample_entities_error"] = str(e)
                
                return collection_info
                
            except Exception as e:
                return {"error": str(e)}
        
        from fastapi.concurrency import run_in_threadpool
        stats = await run_in_threadpool(_get_collection_stats)
        
        status = {
            "vector_store_exists": vector_store is not None,
            "vector_store_type": type(vector_store).__name__ if vector_store else None,
            "initialization_status": "success",
            "collection_stats": stats
        }
        
        return JSONResponse(status)
        
    except Exception as e:
        return JSONResponse({
            "vector_store_exists": vector_store is not None,
            "error": str(e),
            "initialization_status": "failed"
        })

@app.get("/api/debug/space/{space_id}")
async def debug_space_data(space_id: str):
    """Debug endpoint to check what data exists for a specific space"""
    
    try:
        vector_store_instance = await get_vector_store()
        
        def _get_space_data():
            try:
                # Query all entities for this space
                results = vector_store_instance.collection.query(
                    expr=f'space_id == "{space_id}"',
                    limit=100,
                    output_fields=["id", "space_id", "memory_id", "file_name", "content", "chunk_index"]
                )
                
                space_data = {
                    "space_id": space_id,
                    "total_chunks": len(results),
                    "entities": []
                }
                
                for result in results:
                    space_data["entities"].append({
                        "id": result.get("id", ""),
                        "memory_id": result.get("memory_id", ""),
                        "file_name": result.get("file_name", ""),
                        "chunk_index": result.get("chunk_index", ""),
                        "content_preview": str(result.get("content", ""))[:200] + "..."
                    })
                
                return space_data
                
            except Exception as e:
                return {"error": str(e), "space_id": space_id}
        
        from fastapi.concurrency import run_in_threadpool
        data = await run_in_threadpool(_get_space_data)
        
        return JSONResponse(data)
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "space_id": space_id
        })

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the indexed documents using async vector store - optimized for speed"""
    
    start_time = datetime.now()
    
    try:
        # Get vector store instance (should be fast if already initialized)
        vector_store_instance = await get_vector_store()
        
        logger.info(f"🔍 Querying documents in space {request.space_id}")
        logger.info(f"Query: {request.query}")
        logger.info(f"Top K: {request.top_k}")
        
        # Single search call - no threshold iterations
        results = await vector_store_instance.async_search_similar(
            query_text=request.query,
            limit=request.top_k,
            filter_expr=request.filter_expr
        )
        
        if results and len(results) > 0:
            for hit in results[0]:
                results.append({"space_id": hit.entity.get("space_id")})
                
        logger.info(f"✅ Found {len(results)} results")
        
        # Generate AI response efficiently
        ai_response = None
        
        if results and config.GEMINI_KEY:
            logger.info("Generating AI response with Gemini")
            
            # Update query service with current vector store if needed
            if query_service.vector_store is None:
                query_service.vector_store = vector_store_instance
            
            try:
                # Use synchronous method wrapped in thread pool
                def _generate_response():
                    return query_service.generate_response_new(request.query, results)
                
                ai_response = await run_in_threadpool(_generate_response)
                logger.info(f"✅ AI response generated successfully")
                
            except Exception as e:
                logger.error(f"Failed to generate AI response: {str(e)}")
                # Quick fallback without processing all results
                ai_response = f"I found {len(results)} relevant documents but couldn't generate an AI response. Please try again."
                
        elif results:
            # Quick fallback response without AI
            logger.warning("No Gemini API key - providing document excerpts")
            ai_response = f"Found {len(results)} relevant documents from your query. "
            if results:
                ai_response += f"Most relevant excerpt: {results[0].get('content', '')[:200]}..."
        else:
            # No results found
            ai_response = f"No relevant documents found for your query '{request.query}' in this space."
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"⚡ Query completed in {processing_time:.2f}s")
        
        return QueryResponse(
            query=request.query,
            results=results,
            response=ai_response,
            total_results=len(results),
            processing_time=processing_time
        )
        
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    

@app.post("/api/queryOld", response_model=QueryResponse)
async def query_documents_old(request: QueryRequestOld):
    """Query the indexed documents - legacy endpoint"""
    
    start_time = datetime.now()
    
    try:
        # Perform similarity search
        results = vector_store_old.similarity_search(
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
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
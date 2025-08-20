import base64
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import tempfile
import os
import shutil
import logging

from datetime import datetime
from pathlib import Path

from services.document_processor import DocumentProcessor
from services.vector_store import ZillizVectorStore
from services.query_service import QueryService
from services.storage_service import StorageManager
from config import Config
from supabase import create_client, Client


from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
    
@app.post("/api/process-memory")
async def process_memory(
    background_tasks: BackgroundTasks,
    type: str = Form(...),
    fileUrl: str = Form(...),
    fileName: str = Form(...),
    fileType: str = Form(...),
    spaceId: str = Form(...),
    storageMethod: str = Form(...),
    memoryId: str = Form(...)
):
    """Process a file uploaded via Next.js into the RAG system."""
    
    if type != "upload":
        raise HTTPException(status_code=400, detail="Invalid type, must be 'upload'")
    
    if storageMethod not in ["storage", "database"]:
        raise HTTPException(status_code=400, detail="Invalid storage method")
    
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    try:
        # Update status to "processing"
        update_response = supabase.table("memories").update({
            "processing_status": "processing"
        }).eq("id", memoryId).execute()
        
        if not update_response.data:
            logger.warning(f"Failed to update status to processing for memory {memoryId}")
        
        # Retrieve file based on storage method
        temp_path = None
        if storageMethod == "storage":
            try:
                # Extract file_path from public fileUrl
                # fileUrl format: https://project.supabase.co/storage/v1/object/public/uploads/user_id/space_id/timestamp.ext
                if "/uploads/" in fileUrl:
                    file_path = fileUrl.split("/uploads/")[1]
                else:
                    raise HTTPException(status_code=400, detail="Invalid file URL format")
                
                # Download file from Supabase storage
                file_bytes, _ = retrieve_file_from_supabase(
                    config.SUPABASE_URL,
                    config.SUPABASE_KEY,
                    config.SUPABASE_BUCKET_NAME,  # Use config value
                    file_path
                )
                
                # Create temporary file
                temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=Path(fileName).suffix).name
                with open(temp_path, 'wb') as f:
                    f.write(file_bytes)
                    
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to retrieve file from storage: {str(e)}")
                
        elif storageMethod == "database":
            try:
                # fileUrl is base64 data URL
                if fileUrl.startswith("data:"):
                    file_bytes = base64.b64decode(fileUrl.split(',')[1])  # Remove data URL prefix
                else:
                    file_bytes = base64.b64decode(fileUrl)
                
                temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=Path(fileName).suffix).name
                with open(temp_path, 'wb') as f:
                    f.write(file_bytes)
                    
            except (base64.binascii.Error, IndexError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")
        
        if not temp_path:
            raise HTTPException(status_code=500, detail="Failed to retrieve file content")
        
        # Validate file size
        file_size = os.path.getsize(temp_path)
        if file_size > config.MAX_CONTENT_LENGTH:
            os.unlink(temp_path)
            raise HTTPException(status_code=413, detail="File too large")
        
        # Validate file type
        if not document_processor.is_allowed_file(fileName):
            os.unlink(temp_path)
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {document_processor.allowed_extensions}"
            )
        
        # Process document
        processed_data = document_processor.process_document(temp_path)
        
        if not processed_data:
            os.unlink(temp_path)
            raise HTTPException(status_code=500, detail="Failed to process document")
        
        # Enhance metadata
        metadata = processed_data['metadata']
        metadata['space_id'] = spaceId
        metadata['memory_id'] = memoryId
        metadata['file_type'] = fileType
        metadata['original_name'] = Path(fileName).name
        metadata['storage_method'] = storageMethod
        
        # Store in vector store
        document_id = vector_store.store_document(
            filename=fileName,
            chunks=processed_data['content'],
            metadata=metadata
        )
        
        # Clean up
        os.unlink(temp_path)
        
        # Update Supabase status to "completed"
        update_response = supabase.table("memories").update({
            "processing_status": "completed"
        }).eq("id", memoryId).execute()
        
        if not update_response.data:
            logger.error(f"Failed to update status to completed for memory {memoryId}")
        
        return JSONResponse({
            "message": "File processed and added to RAG system successfully",
            "document_id": document_id,
            "space_id": spaceId,
            "memory_id": memoryId,
            "chunks_created": len(processed_data['content'])
        })
        
    except Exception as e:
        logger.error(f"Error processing memory {memoryId}: {str(e)}")
        try:
            supabase.table("memories").update({
                "processing_status": "failed"
            }).eq("id", memoryId).execute()
        except:
            pass  # Don't fail if status update fails
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

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



    
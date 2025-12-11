import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi.concurrency import run_in_threadpool
import uuid

from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, MilvusException
from config import Config

logger = logging.getLogger(__name__)
config = Config()

class ZillizVectorStoreNew:
    """Async vector store implementation using Zilliz Cloud (managed Milvus)."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: Config):
        """Singleton pattern to prevent multiple instances."""
        if cls._instance is None:
            cls._instance = super(ZillizVectorStoreNew, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Config):
        # Only initialize once
        if hasattr(self, '_config_set'):
            return
            
        self.config = config
        self.collection_name = "rag_documents"
        self.embedding_model = None
        self.collection = None
        self._config_set = True
        
        # Validate configuration
        self._validate_config()
        
        logger.info("ZillizVectorStore instance created (lazy initialization)")
    
    def _validate_config(self):
        """Validate Zilliz Cloud configuration."""
        if not self.config.MILVUS_URI:
            raise ValueError("MILVUS_URI is required for Zilliz Cloud connection")
        
        # Check if we have either username/password or API key
        has_credentials = (
            self.config.ZILLIZ_CLOUD_USERNAME and self.config.ZILLIZ_CLOUD_PASSWORD
        ) or self.config.ZILLIZ_CLOUD_API_KEY
        
        if not has_credentials:
            raise ValueError("Either ZILLIZ_CLOUD_USERNAME/PASSWORD or ZILLIZ_CLOUD_API_KEY is required")
        
        logger.info(f"✅ Zilliz Cloud config validated for endpoint: {self.config.MILVUS_URI}")
    
    async def initialize(self):
        """Explicit initialization method to avoid conflicts."""
        if ZillizVectorStoreNew._initialized:
            logger.info("ZillizVectorStore already initialized")
            return
            
        try:
            logger.info("🚀 Starting ZillizVectorStore initialization...")
            
            # Step 1: Initialize embedding model in thread pool
            def _init_embedding_model():
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded successfully")
                return model
            
            self.embedding_model = await run_in_threadpool(_init_embedding_model)
            
            # Step 2: Initialize Zilliz connection in thread pool
            await self._async_connect()
            
            ZillizVectorStoreNew._initialized = True
            logger.info("🚀 ZillizVectorStore fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ZillizVectorStore: {str(e)}")
            raise
    
    async def _async_connect(self):
        """Establish async connection to Zilliz Cloud Serverless."""
        def _connect():
            try:
                # Disconnect existing default connection if it exists
                try:
                    if connections.has_connection("default"):
                        connections.disconnect("default")
                        logger.info("📡 Disconnected existing 'default' connection")
                except Exception as e:
                    logger.warning(f"Warning disconnecting existing connection: {e}")
                
                # For Zilliz Cloud Serverless, use the URI-based connection method
                endpoint = self.config.ZILLIZ_CLOUD_ENDPOINT
                
                # Zilliz Cloud Serverless connection using URI method
                if endpoint.startswith('https://'):
                    zilliz_uri = endpoint
                else:
                    zilliz_uri = f"https://{endpoint}"
                
                logger.info(f"🔗 Connecting to Zilliz Cloud Serverless: {zilliz_uri}")
                logger.info(f"🔑 Using API key authentication")
                
                # Connect using URI method for Zilliz Cloud Serverless
                connections.connect(
                    alias="default",
                    uri=zilliz_uri,
                    token=self.config.ZILLIZ_CLOUD_API_KEY,
                    secure=True
                )
                
                logger.info(f"✅ Connected to Zilliz Cloud Serverless: {zilliz_uri}")
                
                # Create or get collection
                self._create_collection_if_not_exists()
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Zilliz connection failed: {str(e)}")
                logger.error(f"Connection URI used: {zilliz_uri if 'zilliz_uri' in locals() else 'Not created'}")
                logger.error(f"API Key provided: {'Yes' if self.config.ZILLIZ_CLOUD_API_KEY else 'No'}")
                raise
        
        await run_in_threadpool(_connect)
    
    def _create_collection_if_not_exists(self):
        """Create collection schema if it doesn't exist."""
        try:
            from pymilvus import utility
            
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"📁 Using existing collection: {self.collection_name}")
                
                # Load collection to ensure it's ready
                try:
                    self.collection.load()
                    logger.info(f"📋 Collection {self.collection_name} loaded successfully")
                except Exception as e:
                    logger.warning(f"Warning loading collection: {e}")
                return
            
            logger.info(f"📁 Creating new collection: {self.collection_name}")
            
            # Define collection schema with appropriate field sizes
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=200, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # all-MiniLM-L6-v2 dim
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="space_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="memory_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="file_size", dtype=DataType.INT64),
                FieldSchema(name="storage_method", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="file_url", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="total_chunks", dtype=DataType.INT64),
                FieldSchema(name="char_count", dtype=DataType.INT64),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="RAG Knowledge Base Documents",
                enable_dynamic_field=False
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            # Create index for vector field - use AUTOINDEX for Zilliz Cloud Serverless
            index_params = {
                "metric_type": "COSINE",
                "index_type": "AUTOINDEX",  # AUTOINDEX is recommended for Zilliz Cloud
                "params": {}
            }
            
            logger.info("📊 Creating vector index...")
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            # Load collection after index creation
            logger.info("📋 Loading collection...")
            self.collection.load()
            
            logger.info(f"✅ Created and loaded collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection: {str(e)}")
            raise
    
    async def _ensure_initialized(self):
        """Ensure the vector store is initialized before operations."""
        if not ZillizVectorStoreNew._initialized:
            logger.info("⏳ Initializing ZillizVectorStore...")
            await self.initialize()
    
    async def async_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts asynchronously."""
        await self._ensure_initialized()
        
        def _generate_embeddings():
            try:
                # Generate embeddings using sentence transformers
                embeddings = self.embedding_model.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=32  # Process in batches for efficiency
                )
                return embeddings.tolist()
                
            except Exception as e:
                logger.error(f"❌ Embedding generation failed: {str(e)}")
                raise
        
        logger.info(f"🧠 Generating embeddings for {len(texts)} text chunks...")
        embeddings = await run_in_threadpool(_generate_embeddings)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        
        return embeddings
    async def _async_insert_entities_batch(self, entities: List[Dict[str, Any]]) -> bool:
        """Insert entities in smaller batches or one by one if needed."""
        def _insert_batch():
            try:
                # Ensure collection is loaded
                try:
                    logger.info("📋 Loading collection for batch insertion...")
                    self.collection.load()
                except Exception as load_error:
                    logger.warning(f"Collection load warning: {load_error}")
                
                # Try batch insertion first (smaller batches)
                batch_size = 5  # Smaller batch size for testing
                total_inserted = 0
                
                for i in range(0, len(entities), batch_size):
                    batch = entities[i:i + batch_size]
                    logger.info(f"💾 Inserting batch {i//batch_size + 1}: {len(batch)} entities")
                    
                    # Prepare batch data
                    batch_data = [
                        [entity["id"] for entity in batch],
                        [entity["embedding"] for entity in batch],
                        [entity["content"] for entity in batch],
                        [entity["space_id"] for entity in batch],
                        [entity["memory_id"] for entity in batch],
                        [entity["file_name"] for entity in batch],
                        [entity["file_type"] for entity in batch],
                        [entity["file_size"] for entity in batch],
                        [entity["storage_method"] for entity in batch],
                        [entity["file_url"] for entity in batch],
                        [entity["chunk_index"] for entity in batch],
                        [entity["total_chunks"] for entity in batch],
                        [entity["char_count"] for entity in batch],
                        [entity["created_at"] for entity in batch],
                    ]
                    
                    # Insert batch
                    insert_result = self.collection.insert(batch_data)
                    total_inserted += len(batch)
                    
                    logger.info(f"✅ Batch inserted successfully: {len(batch)} entities")
                
                # Flush all data
                logger.info("🔄 Flushing all batched data...")
                self.collection.flush()
                
                logger.info(f"💾 Successfully inserted {total_inserted} entities in batches")
                return True
                
            except Exception as e:
                logger.error(f"❌ Batch insertion failed: {str(e)}")
                
                # If batch fails, try one by one
                logger.info("🔄 Trying single entity insertion...")
                return self._insert_one_by_one(entities)
        
        return await run_in_threadpool(_insert_batch)
    
    def _insert_one_by_one(self, entities: List[Dict[str, Any]]) -> bool:
        """Insert entities one by one as a last resort."""
        try:
            success_count = 0
            
            for i, entity in enumerate(entities):
                try:
                    # Prepare single entity data
                    single_data = [
                        [entity["id"]],
                        [entity["embedding"]],
                        [entity["content"]],
                        [entity["space_id"]],
                        [entity["memory_id"]],
                        [entity["file_name"]],
                        [entity["file_type"]],
                        [entity["file_size"]],
                        [entity["storage_method"]],
                        [entity["file_url"]],
                        [entity["chunk_index"]],
                        [entity["total_chunks"]],
                        [entity["char_count"]],
                        [entity["created_at"]],
                    ]
                    
                    # Insert single entity
                    self.collection.insert(single_data)
                    success_count += 1
                    
                    if (i + 1) % 5 == 0:  # Log progress every 5 entities
                        logger.info(f"Inserted {i + 1}/{len(entities)} entities")
                        
                except Exception as e:
                    logger.error(f"Failed to insert entity {i}: {str(e)}")
            
            # Flush all data
            self.collection.flush()
            
            logger.info(f"💾 Single insertion completed: {success_count}/{len(entities)} entities")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Single entity insertion failed: {str(e)}")
            return False
    
    async def async_add_documents(
        self,
        chunks: List[Dict[str, Any]],
        space_id: str,
        memory_id: str,
        file_name: str,
        file_type: str,
        file_size: int,
        storage_method: str,
        file_url: str
    ) -> bool:
        """Add document chunks to the vector store asynchronously."""
        try:
            await self._ensure_initialized()
            
            if not chunks:
                logger.warning("No chunks provided for vector storage")
                return False
            
            logger.info(f"📥 Processing {len(chunks)} chunks for vector storage...")
            
            # Extract text content from chunks
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings asynchronously
            embeddings = await self.async_generate_embeddings(texts)
            
            # Prepare data for insertion with proper validation
            entities = []
            current_time = datetime.now().isoformat()[:50]
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate unique ID for each chunk
                chunk_id = f"{memory_id[:36]}_{chunk['metadata']['chunk_index']:04d}"
                
                entity = {
                    "id": str(chunk_id)[:200],
                    "embedding": embedding,
                    "content": str(chunk['content'])[:65535],
                    "space_id": str(space_id)[:100],
                    "memory_id": str(memory_id)[:100],
                    "file_name": str(file_name)[:500],
                    "file_type": str(file_type)[:50],
                    "file_size": int(file_size),
                    "storage_method": str(storage_method)[:50],
                    "file_url": str(file_url)[:2000],
                    "chunk_index": int(chunk['metadata']['chunk_index']),
                    "total_chunks": int(chunk['metadata']['total_chunks']),
                    "char_count": int(chunk['metadata']['char_count']),
                    "created_at": str(current_time)[:50]
                }
                
                entities.append(entity)
            
            logger.info(f"📊 Prepared {len(entities)} entities for insertion")
            
            # Try the new batch insertion method first
            success = await self._async_insert_entities_batch(entities)
            
            if success:
                logger.info(f"✅ Successfully stored {len(entities)} chunks in vector database")
                return True
            else:
                logger.error("❌ Failed to store chunks in vector database")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error adding documents to vector store: {str(e)}", exc_info=True)
            return False
    
    async def _async_insert_entities(self, entities: List[Dict[str, Any]]) -> bool:
        """Insert entities into Zilliz collection asynchronously."""
        def _insert():
            try:
                # Ensure collection is loaded - use a safer approach
                try:
                    # Try to load the collection (this is safe to call multiple times)
                    logger.info("📋 Loading collection for insertion...")
                    self.collection.load()
                except Exception as load_error:
                    logger.warning(f"Collection load warning (may already be loaded): {load_error}")
                
                # For Zilliz Cloud Serverless, use the list-of-lists format instead of dict format
                # This is the alternative format that sometimes works better with cloud instances
                logger.info(f"💾 Inserting {len(entities)} entities using list format...")
                
                # Prepare data as list of lists in the order of the schema fields
                data = [
                    [entity["id"] for entity in entities],           # id field
                    [entity["embedding"] for entity in entities],   # embedding field
                    [entity["content"] for entity in entities],     # content field
                    [entity["space_id"] for entity in entities],    # space_id field
                    [entity["memory_id"] for entity in entities],   # memory_id field
                    [entity["file_name"] for entity in entities],   # file_name field
                    [entity["file_type"] for entity in entities],   # file_type field
                    [entity["file_size"] for entity in entities],   # file_size field
                    [entity["storage_method"] for entity in entities], # storage_method field
                    [entity["file_url"] for entity in entities],    # file_url field
                    [entity["chunk_index"] for entity in entities], # chunk_index field
                    [entity["total_chunks"] for entity in entities], # total_chunks field
                    [entity["char_count"] for entity in entities],  # char_count field
                    [entity["created_at"] for entity in entities],  # created_at field
                ]
                
                # Log data structure for debugging
                logger.info(f"Data format: List of {len(data)} lists")
                logger.info(f"First list (id) length: {len(data[0])}")
                logger.info(f"Sample ID: {data[0][0] if data[0] else 'None'}")
                logger.info(f"Embedding dimension: {len(data[1][0]) if data[1] and data[1][0] else 'None'}")
                
                # Validate data before insertion
                for i, field_data in enumerate(data):
                    if not field_data:
                        logger.error(f"Empty data for field index: {i}")
                        return False
                    logger.info(f"Field {i}: {len(field_data)} items, sample type: {type(field_data[0]) if field_data else 'None'}")
                
                # Insert data using the list format
                insert_result = self.collection.insert(data)
                
                # Flush to ensure data is persisted
                logger.info("🔄 Flushing data to storage...")
                self.collection.flush()
                
                logger.info(f"💾 Successfully inserted {len(entities)} entities into collection")
                logger.info(f"Insert result: {insert_result}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Milvus insertion error: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                
                # If list format fails, try the alternative approach with explicit field names
                logger.info("🔄 Trying alternative insertion method...")
                try:
                    return self._insert_with_field_names(entities)
                except Exception as e2:
                    logger.error(f"❌ Alternative insertion also failed: {str(e2)}")
                    return False
        
        return await run_in_threadpool(_insert)
    
    def _insert_with_field_names(self, entities: List[Dict[str, Any]]) -> bool:
        """Alternative insertion method using field names explicitly."""
        try:
            # Get the schema field names in order
            schema_fields = [field.name for field in self.collection.schema.fields]
            logger.info(f"Schema fields in order: {schema_fields}")
            
            # Prepare data using field names
            field_data = {}
            for field_name in schema_fields:
                if field_name in entities[0]:  # Check if field exists in entity
                    field_data[field_name] = [entity[field_name] for entity in entities]
                else:
                    logger.error(f"Field {field_name} not found in entity data")
                    return False
            
            logger.info(f"Prepared field data with keys: {list(field_data.keys())}")
            
            # Insert using field names
            insert_result = self.collection.insert(field_data)
            self.collection.flush()
            
            logger.info(f"💾 Alternative method successfully inserted {len(entities)} entities")
            return True
            
        except Exception as e:
            logger.error(f"❌ Alternative insertion method failed: {str(e)}")
            return False
    

    async def async_search_similar(
        self,
        query_text: str,
        limit: int = 10,
        filter_expr: Optional[str] = None,
        vector_field: str = "embedding",  # 1. Added this parameter
        **kwargs                       # 2. Added kwargs to catch any other unexpected arguments safely
    ) -> List[Dict[str, Any]]:
        """Search for similar documents asynchronously."""
        try:
            await self._ensure_initialized()
            
            # Generate query embedding
            query_embeddings = await self.async_generate_embeddings([query_text])
            query_embedding = query_embeddings[0] 
            
            def _search():
                try:
                    # Ensure collection is loaded for search
                    self.collection.load()
                    
                    # Search parameters for AUTOINDEX
                    search_params = {
                        "metric_type": "COSINE",
                        "params": {"nprobe": 10}
                    }
                    
                    logger.info(f"🔍 Searching with query embedding dimension: {len(query_embedding)}")
                    
                    # Perform search using the passed vector_field
                    results = self.collection.search(
                        data=[query_embedding], 
                        anns_field=vector_field,     # 3. Use the variable here instead of hardcoded "vector"
                        param=search_params,
                        limit=limit,
                        output_fields=["content", "file_name", "chunk_index", "total_chunks", "memory_id", "file_url", "space_id"], 
                        expr=filter_expr
                    )
                    
                    logger.info(f"🔍 Search returned {len(results[0]) if results and len(results) > 0 else 0} results")
                    
                    # Process results efficiently
                    similar_docs = []
                    if results and len(results) > 0:
                        for hit in results[0]:
                            similar_docs.append({
                                "id": hit.id,
                                "content": hit.entity.get("content"),
                                "file_name": hit.entity.get("file_name"),
                                "chunk_index": hit.entity.get("chunk_index"),
                                "total_chunks": hit.entity.get("total_chunks"),
                                "memory_id": hit.entity.get("memory_id"),
                                "file_url": hit.entity.get("file_url"),
                                "space_id": hit.entity.get("space_id"),
                                "similarity_score": float(hit.score)
                            })
                    
                    logger.info(f"🎯 Processed {len(similar_docs)} results")
                    return similar_docs
                    
                except Exception as e:
                    logger.error(f"❌ Search error: {str(e)}")
                    return []
            
            results = await run_in_threadpool(_search)
            logger.info(f"✅ Search completed: {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in async search: {str(e)}")
            return []
        
        
    async def async_delete_memory_documents(self, memory_id: str) -> bool:
        """Delete all documents associated with a memory ID."""
        try:
            await self._ensure_initialized()
            
            def _delete():
                try:
                    # Ensure collection is loaded for deletion
                    try:
                        self.collection.load()
                    except Exception as load_error:
                        logger.warning(f"Collection load warning during delete: {load_error}")
                    
                    # Delete by memory_id
                    expr = f'memory_id == "{memory_id}"'
                    delete_result = self.collection.delete(expr)
                    
                    # Flush changes
                    self.collection.flush()
                    
                    logger.info(f"🗑️ Deleted documents for memory {memory_id}")
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Delete error: {str(e)}")
                    return False
            
            return await run_in_threadpool(_delete)
            
        except Exception as e:
            logger.error(f"❌ Error deleting memory documents: {str(e)}")
            return False
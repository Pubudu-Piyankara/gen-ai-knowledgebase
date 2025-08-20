import logging
import uuid
import json
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from datetime import datetime

logger = logging.getLogger(__name__)

class ZillizVectorStore:
    """Vector store using Zilliz/Milvus for scalable similarity search"""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: str = "19530",
                 uri: str = None,
                 token: str = None,
                 collection_name: str = "document_embeddings",
                 dimension: int = 384):
        
        self.host = host
        self.port = port
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dimension = dimension
        self.collection = None
        self._connected = False
        
        # Determine connection type
        self.use_cloud = bool(uri and token)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Don't connect immediately - use lazy connection
        if self.use_cloud:
            logger.info(f"ZillizVectorStore initialized for Zilliz Cloud: {self.uri}")
        else:
            logger.info(f"ZillizVectorStore initialized for localhost: {self.host}:{self.port}")
    
    def _ensure_connected(self):
        """Ensure connection to Milvus/Zilliz exists"""
        if not self._connected:
            try:
                self._connect()
                self._create_collection()
            except Exception as e:
                logger.error(f"Failed to establish Milvus connection: {str(e)}")
                self._connected = False
                raise
    
    def _connect(self):
        """Connect to Milvus/Zilliz"""
        try:
            if self.use_cloud:
                # Connect to Zilliz Cloud
                connections.connect(
                    alias="default",
                    uri=self.uri,
                    token=self.token
                )
                logger.info(f"Connected to Zilliz Cloud: {self.uri}")
            else:
                # Connect to local Milvus
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port
                )
                logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            
            self._connected = True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def _create_collection(self):
        """Create collection with optimized schema"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
            
            # Log the existing schema to debug
            schema = self.collection.schema
            logger.info(f"Existing collection schema fields:")
            for field in schema.fields:
                logger.info(f"  - {field.name}: {field.dtype}")
            return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),  # Changed from "embedding" to "vector"
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Document embeddings for RAG system"
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # Create index for vector search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        self.collection.create_index(
            field_name="vector",  # Changed from "embedding" to "vector"
            index_params=index_params
        )
        
        logger.info(f"Created collection: {self.collection_name}")
    
    def store_document(self, 
                      filename: str, 
                      chunks: List[Dict[str, Any]], 
                      metadata: Dict[str, Any]) -> str:
        """Store document chunks with embeddings"""
        
        try:
            self._ensure_connected()  # Ensure connection before using
        except Exception as e:
            logger.error(f"Cannot store document - Milvus not available: {str(e)}")
            # Return a fake document ID for now - in production you might want to queue this
            return f"offline_{str(uuid.uuid4())}"
        
        document_id = str(uuid.uuid4())
        
        try:
            # Extract text from chunks for embedding
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings in batch
            chunk_embeddings = self.embedding_model.encode(
                chunk_texts,
                convert_to_tensor=False,
                show_progress_bar=True
            )
            
            # Prepare insertion data as list of dictionaries
            entities = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                chunk_id = str(uuid.uuid4())
                
                # Combine document metadata with chunk metadata
                combined_metadata = {
                    **metadata,
                    **chunk.get('metadata', {}),
                    'document_metadata': metadata
                }
                
                entity = {
                    "id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk.get('chunk_index', i),
                    "chunk_type": chunk.get('chunk_type', 'text'),
                    "text": chunk['text'],
                    "vector": embedding.tolist(),  # Changed from "embedding" to "vector"
                    "metadata": combined_metadata,
                    "created_at": int(datetime.now().timestamp())
                }
                entities.append(entity)
            
            # Insert data
            self.collection.insert(entities)
            self.collection.flush()
            
            logger.info(f"Stored document {document_id} with {len(chunks)} chunks")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         top_k: int = 5,
                         filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform similarity search with optional filtering"""
        
        try:
            self._ensure_connected()  # Ensure connection before using
        except Exception as e:
            logger.error(f"Cannot perform search - Milvus not available: {str(e)}")
            return []  # Return empty results when not connected
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)[0]
            
            # Load collection
            self.collection.load()
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="vector",  # Changed from "embedding" to "vector"
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "chunk_index", "chunk_type", "text", "metadata"],
                expr=filter_expr
            )
            
            # Format results
            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    'id': hit.id,
                    'document_id': hit.entity.get('document_id'),
                    'chunk_index': hit.entity.get('chunk_index'),
                    'chunk_type': hit.entity.get('chunk_type'),
                    'text': hit.entity.get('text'),
                    'metadata': hit.entity.get('metadata'),
                    'similarity_score': float(hit.score)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            self._ensure_connected()  # Ensure connection before using
        except Exception as e:
            logger.error(f"Cannot delete document - Milvus not available: {str(e)}")
            return False  # Return False when not connected
        
        try:
            expr = f'document_id == "{document_id}"'
            self.collection.delete(expr)
            self.collection.flush()
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            self._ensure_connected()  # Ensure connection before using
            self.collection.load()
            stats = {
                'total_entities': self.collection.num_entities,
                'collection_name': self.collection_name,
                'dimension': self.dimension,
                'status': 'connected'
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                'total_entities': 0,
                'collection_name': self.collection_name,
                'dimension': self.dimension,
                'status': 'disconnected',
                'error': str(e)
            }
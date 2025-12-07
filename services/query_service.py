import logging
from typing import Dict, List, Any
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

class QueryService:
    """Handles query processing and response generation"""
    
    def __init__(self, vector_store, GEMINI_KEY=None):
        self.vector_store = vector_store
        self.GEMINI_KEY = GEMINI_KEY or os.getenv('GEMINI_KEY')
        
                # Initialize Gemini client
        if self.GEMINI_KEY:
            genai.configure(api_key=self.GEMINI_KEY)
            # Use the updated model name - gemini-1.5-flash is faster and more cost-effective
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
            logger.info("Gemini client initialized successfully with gemini-1.5-flash model")
        else:
            self.gemini_model = None
            logger.warning("No Gemini API key provided. Response generation will be disabled.")
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Process query and return results with optional AI-generated response"""
        # Perform similarity search
        matches = self.vector_store.similarity_search(query_text, top_k=top_k)
        
        response = ""
        if self.GEMINI_KEY and matches:
            response = self._generate_response(query_text, matches)
        
        return {
            'matches': matches,
            'response': response
        }
    
    def generate_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Public method to generate AI response from query and results"""
        if not self.GEMINI_KEY:
            return "AI response generation is not available (OpenAI API key not configured)."
        
        if not results:
            return "I couldn't find any relevant information in the documents to answer your question."
        
        return self._generate_response(query, results)
    
    def _generate_response(self, query: str, matches: List[Dict[str, Any]]) -> str:
        """Generate AI response using retrieved context"""
        try:
            # Check if Gemini client is available
            if not self.gemini_model:
                logger.warning("Gemini model not available")
                return "I found relevant documents but AI response generation is not available."
            
            # Prepare context from matches
            context_parts = []
            for match in matches[:3]:  # Use top 3 matches
                content = match.get('content', match.get('text', ''))
                filename = match.get('metadata', {}).get('filename', 'Unknown')
                if content:
                    context_parts.append(f"Document: {filename}\nContent: {content}")
            
            context = "\n\n".join(context_parts)
            
            if not context.strip():
                logger.warning("No context extracted from matches")
                return "I found some documents but couldn't extract meaningful content to answer your question."

            prompt = f"""Based on the following context from uploaded documents, answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Please provide a helpful answer based on the context above:"""

            logger.info(f"Generating response with Gemini for query: {query}")
            response = self.gemini_model.generate_content(prompt)
            
            if response and hasattr(response, 'text') and response.text:
                logger.info("Successfully generated response with Gemini")
                return response.text.strip()
            else:
                logger.error(f"Gemini response has no text content: {response}")
                return "I found relevant documents but couldn't generate a complete response."
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return "I found relevant documents but couldn't generate a response due to an error."
        
    def generate_response_new(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Public method to generate AI response from query and results - optimized"""
        if not self.GEMINI_KEY:
            return "AI response generation is not available (Gemini API key not configured)."
        
        if not results:
            return "I couldn't find any relevant information in the documents to answer your question."
        
        return self._generate_response_new(query, results)


    def _generate_response_new(self, query: str, matches: List[Dict[str, Any]]) -> str:
        """Generate AI response using retrieved context - optimized"""
        try:
            # Check if Gemini client is available
            if not self.gemini_model:
                logger.warning("Gemini model not available")
                return "I found relevant documents but AI response generation is not available."
            
            # Debug: Log the matches structure
            logger.info(f"Processing {len(matches)} matches for query: {query}")
            for i, match in enumerate(matches[:3]):
                logger.info(f"Match {i+1}: {list(match.keys())}")
                content = match.get('content', match.get('text', ''))
                logger.info(f"Match {i+1} content preview: {content[:100] if content else 'NO CONTENT'}")
            
            # Prepare context from top 3 matches only (faster processing)
            context_parts = []
            for i, match in enumerate(matches[:3]):
                # Try multiple possible content field names
                content = match.get('content') or match.get('text') or match.get('page_content') or ""
                filename = match.get('file_name') or match.get('metadata', {}).get('filename', 'Document')
                
                if content and len(content.strip()) > 0:
                    # Limit content length for faster processing but keep enough context
                    content_preview = content[:1500]  # Increased from 1000 to 1500 chars
                    context_parts.append(f"Source: {filename}\nContent: {content_preview}")
                    logger.info(f"Added context part {i+1}: {len(content_preview)} chars from {filename}")
                else:
                    logger.warning(f"Match {i+1} has no content: {match}")
            
            if not context_parts:
                logger.error("No content extracted from any matches")
                return "I found some documents but couldn't extract meaningful content to answer your question."
            
            context = "\n\n---\n\n".join(context_parts)
            
            logger.info(f"Final context length: {len(context)} characters")
            logger.info(f"Context preview: {context[:200]}...")

            # More explicit prompt for better response
            prompt = f"""You are a helpful AI assistant. Based on the following document content, answer the user's question accurately and informatively.

    IMPORTANT: Use ONLY the information provided in the context below. If the context contains relevant information about the question, provide a comprehensive answer. If the context doesn't contain relevant information, clearly state that.

    Context from documents:
    {context}

    User Question: {query}

    Please provide a helpful and accurate answer based on the context above. If the context contains information relevant to the question, explain it clearly. If not, say so explicitly."""

            logger.info(f"Sending prompt to Gemini (length: {len(prompt)} chars)")
            
            response = self.gemini_model.generate_content(prompt)
            
            if response and hasattr(response, 'text') and response.text:
                response_text = response.text.strip()
                logger.info(f"Gemini response received: {response_text[:100]}...")
                return response_text
            else:
                logger.error(f"Gemini response has no text content: {response}")
                return "I found relevant documents but couldn't generate a complete response."
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return "I found relevant documents but encountered an error generating the response."


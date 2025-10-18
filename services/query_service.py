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
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
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
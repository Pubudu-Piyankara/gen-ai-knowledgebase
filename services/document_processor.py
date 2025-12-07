import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import tempfile
from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

logger = logging.getLogger(__name__)

class BasicDocumentProcessor:
    """Enhanced document processor using DocLing for better parsing"""
    
    def __init__(self):
        # Initialize DocLing converter with optimized settings
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for scanned documents
        pipeline_options.do_table_structure = True  # Extract table structure
        
        self.converter = DocumentConverter(
            format_options={
                PdfFormatOption: pipeline_options
            }
        )
        
        self.allowed_extensions = {
            '.pdf', '.docx', '.doc', '.pptx', '.ppt', 
            '.txt', '.md', '.html', '.rtf'
        }
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        return Path(filename).suffix.lower() in self.allowed_extensions
    
    def process_document(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Process document using DocLing and extract structured content"""
        try:
            if not self.is_allowed_file(filepath):
                logger.error(f"Unsupported file type: {Path(filepath).suffix}")
                return None
            
            # Convert document using DocLing
            result: ConversionResult = self.converter.convert(filepath)
            
            if not result.document:
                logger.error(f"Failed to convert document: {filepath}")
                return None
            
            # Extract text content
            try:
                if hasattr(result.document, 'export_to_text'):
                    full_text = result.document.export_to_text()
                elif hasattr(result.document, 'text'):
                    full_text = result.document.text
                elif hasattr(result.document, 'to_text'):
                    full_text = result.document.to_text()
                else:
                    full_text = str(result.document)
            except Exception as e:
                logger.warning(f"Failed to extract text, using fallback method: {e}")
                full_text = str(result.document)
            
            # Extract metadata
            metadata = self._extract_metadata(result, filepath)
            
            # Extract structured elements
            structured_content = self._extract_structured_content(result)
            
            # Create chunks with context preservation
            chunks = self._create_intelligent_chunks(
                full_text, 
                structured_content,
                metadata
            )
            
            return {
                'content': chunks,
                'metadata': metadata,
                'structured_content': structured_content,
                'total_length': len(full_text),
                'document_type': self._identify_document_type(result)
            }
        
        except Exception as e:
            logger.error(f"Error processing document {filepath}: {str(e)}")
            return None
    
    def _extract_metadata(self, result: ConversionResult, filepath: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from document"""
        stat = os.stat(filepath)
        
        metadata = {
            'filename': Path(filepath).name,
            'file_size': stat.st_size,
            'created_at': stat.st_ctime,
            'modified_at': stat.st_mtime,
            'file_type': Path(filepath).suffix.lower(),
            'page_count': len(result.document.pages) if result.document.pages else 0,
            'language': getattr(result.document, 'language', 'unknown'),
            'title': getattr(result.document, 'title', ''),
            'author': getattr(result.document, 'author', ''),
            'creation_date': getattr(result.document, 'creation_date', None),
        }
        
        return metadata
    
    def _extract_structured_content(self, result: ConversionResult) -> Dict[str, List]:
        """Extract structured elements like tables, images, headers"""
        structured = {
            'tables': [],
            'images': [],
            'headers': [],
            'lists': [],
            'footnotes': []
        }
        
        if not result.document:
            return structured
        
        # Extract tables
        tables = getattr(result.document, 'tables', [])
        for table in tables:
            try:
                # Try different methods to extract table text
                table_text = ""
                if hasattr(table, 'export_to_text'):
                    table_text = table.export_to_text()
                elif hasattr(table, 'text'):
                    table_text = table.text
                elif hasattr(table, 'to_text'):
                    table_text = table.to_text()
                elif hasattr(table, '__str__'):
                    table_text = str(table)
                else:
                    table_text = f"Table with {len(getattr(table, 'cells', []))} cells"
                
                structured['tables'].append({
                    'content': table_text,
                    'location': getattr(table, 'bbox', None),
                    'page': getattr(table, 'page', None)
                })
            except Exception as e:
                logger.warning(f"Failed to extract table content: {e}")
                structured['tables'].append({
                    'content': "Table content extraction failed",
                    'location': getattr(table, 'bbox', None),
                    'page': getattr(table, 'page', None)
                })
        
        # Extract figures/images
        for figure in getattr(result.document, 'figures', []):
            structured['images'].append({
                'caption': getattr(figure, 'caption', ''),
                'location': getattr(figure, 'bbox', None),
                'page': getattr(figure, 'page', None)
            })
        
        return structured
    
    def _identify_document_type(self, result: ConversionResult) -> str:
        """Identify document type based on content analysis"""
        if not result.document:
            return 'unknown'
        
        # Simple heuristics for document type identification
        try:
            if hasattr(result.document, 'export_to_text'):
                text = result.document.export_to_text().lower()
            elif hasattr(result.document, 'text'):
                text = result.document.text.lower()
            else:
                text = str(result.document).lower()
        except Exception as e:
            logger.warning(f"Failed to extract text for document type identification: {e}")
            text = ""
        
        if any(keyword in text for keyword in ['abstract', 'introduction', 'methodology', 'conclusion']):
            return 'academic_paper'
        elif any(keyword in text for keyword in ['agenda', 'meeting', 'minutes']):
            return 'meeting_notes'
        elif len(getattr(result.document, 'tables', [])) > 3:
            return 'report'
        else:
            return 'document'
    
    def _create_intelligent_chunks(
        self, 
        text: str, 
        structured_content: Dict[str, List],
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks"""
        
        chunks = []
        
        # Split text into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_size = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If adding this paragraph exceeds chunk size, finalize current chunk
            if current_size + para_size > chunk_size and current_chunk:
                chunk_data = {
                    'text': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'chunk_type': 'text',
                    'metadata': {
                        'source_file': metadata['filename'],
                        'chunk_size': len(current_chunk),
                        'file_type': metadata['file_type']
                    }
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + para
                else:
                    current_chunk = para
                current_size = len(current_chunk)
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_size += para_size
        
        # Add final chunk
        if current_chunk.strip():
            chunk_data = {
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'chunk_type': 'text',
                'metadata': {
                    'source_file': metadata['filename'],
                    'chunk_size': len(current_chunk),
                    'file_type': metadata['file_type']
                }
            }
            chunks.append(chunk_data)
        
        # Add structured content as separate chunks
        for table_idx, table in enumerate(structured_content['tables']):
            chunks.append({
                'text': f"Table {table_idx + 1}:\n{table['content']}",
                'chunk_index': len(chunks),
                'chunk_type': 'table',
                'metadata': {
                    'source_file': metadata['filename'],
                    'table_index': table_idx,
                    'file_type': metadata['file_type']
                }
            })
        
        return chunks
import os
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from datetime import datetime

import io
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from config import Config

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available. PDF processing disabled.")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available. DOCX processing disabled.")

try:
    import markdown
    from bs4 import BeautifulSoup
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    logger.warning("markdown/beautifulsoup4 not available. Markdown processing will use plain text fallback.")

try:
    from striprtf.striprtf import rtf_to_text
    RTF_AVAILABLE = True
except ImportError:
    RTF_AVAILABLE = False
    logger.warning("striprtf not available. RTF processing disabled.")

class DocumentProcessor:
    """Document processor for extracting text content from various file formats."""
    
    def __init__(self):
        self.config = Config()
        # Build supported extensions based on available dependencies
        base_extensions = {'.txt', '.md'}  # Always supported
        
        if PDF_AVAILABLE:
            base_extensions.add('.pdf')
        if DOCX_AVAILABLE:
            base_extensions.update({'.docx', '.doc'})
        if RTF_AVAILABLE:
            base_extensions.add('.rtf')
        
        # ODT is supported through manual parsing (no external dependency needed)
        base_extensions.add('.odt')
        
        self.supported_extensions = base_extensions
        logger.info(f"DocumentProcessor initialized with support for: {sorted(self.supported_extensions)}")
    
    def process_file(self, file_path: str, original_filename: str = None) -> str:
        """
        Process a file and extract its content.
        
        Args:
            file_path: Path to the temporary file
            original_filename: Original filename to determine file type
        
        Returns:
            Extracted text content
        """
        try:
            # Use original filename for type detection if provided
            filename_to_check = original_filename if original_filename else file_path
            file_extension = os.path.splitext(filename_to_check)[1].lower()
            
            logger.info(f"📄 Processing file type: {file_extension} (original: {original_filename})")
            
            if file_extension == '.pdf':
                if not PDF_AVAILABLE:
                    raise ValueError("PDF processing not available. Install PyPDF2: pip install PyPDF2")
                return self._process_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                if not DOCX_AVAILABLE:
                    raise ValueError("DOCX processing not available. Install python-docx: pip install python-docx")
                return self._process_docx(file_path)
            elif file_extension == '.txt':
                return self._process_txt(file_path)
            elif file_extension == '.md':
                return self._process_markdown(file_path)
            elif file_extension == '.rtf':
                if not RTF_AVAILABLE:
                    raise ValueError("RTF processing not available. Install striprtf: pip install striprtf")
                return self._process_rtf(file_path)
            elif file_extension == '.odt':
                return self._process_odt(file_path)
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                raise ValueError(f"Unsupported file type: {file_extension}. Supported: {sorted(self.supported_extensions)}")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        try:
            text_content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Page {page_num + 1} ---\n"
                            text_content += page_text
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            if not text_content.strip():
                logger.warning("No text extracted from PDF, file might be image-based")
                return "No text content could be extracted from this PDF file."
            
            logger.info(f"✅ Extracted {len(text_content)} characters from PDF")
            return self._clean_text(text_content)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _process_docx(self, file_path: str) -> str:
        """Extract text from DOCX files."""
        try:
            doc = Document(file_path)
            text_content = ""
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_content += " | ".join(row_text) + "\n"
            
            if not text_content.strip():
                logger.warning("No text extracted from DOCX file")
                return "No text content could be extracted from this DOCX file."
            
            logger.info(f"✅ Extracted {len(text_content)} characters from DOCX")
            return self._clean_text(text_content)
            
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            raise
    
    def _process_txt(self, file_path: str) -> str:
        """Extract text from TXT files."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        
                    logger.info(f"✅ Read TXT file with {encoding} encoding: {len(content)} characters")
                    return self._clean_text(content)
                    
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try binary mode and decode with errors='ignore'
            with open(file_path, 'rb') as file:
                content = file.read().decode('utf-8', errors='ignore')
                
            logger.info(f"✅ Read TXT file with error handling: {len(content)} characters")
            return self._clean_text(content)
            
        except Exception as e:
            logger.error(f"Error processing TXT file: {str(e)}")
            raise
    
    def _process_markdown(self, file_path: str) -> str:
        """Extract text from Markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
            
            if MARKDOWN_AVAILABLE:
                # Convert markdown to HTML then extract text
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                text_content = soup.get_text()
                logger.info(f"✅ Extracted {len(text_content)} characters from Markdown (with formatting)")
            else:
                # Fallback: treat as plain text but remove common markdown syntax
                text_content = self._clean_markdown_syntax(md_content)
                logger.info(f"✅ Extracted {len(text_content)} characters from Markdown (plain text fallback)")
            
            return self._clean_text(text_content)
            
        except Exception as e:
            logger.error(f"Error processing Markdown: {str(e)}")
            # Fallback: treat as plain text
            return self._process_txt(file_path)
    
    def _process_rtf(self, file_path: str) -> str:
        """Extract text from RTF files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                rtf_content = file.read()
            
            text_content = rtf_to_text(rtf_content)
            
            logger.info(f"✅ Extracted {len(text_content)} characters from RTF")
            return self._clean_text(text_content)
            
        except Exception as e:
            logger.error(f"Error processing RTF: {str(e)}")
            raise
    
    def _process_odt(self, file_path: str) -> str:
        """Extract text from ODT files using manual XML parsing."""
        try:
            text_content = ""
            
            with zipfile.ZipFile(file_path, 'r') as odt_file:
                # ODT files contain content.xml with the actual text
                try:
                    content_xml = odt_file.read('content.xml')
                    
                    # Parse XML and extract text
                    root = ET.fromstring(content_xml)
                    
                    # Remove namespace prefixes for easier parsing
                    for elem in root.iter():
                        if elem.tag.startswith('{'):
                            elem.tag = elem.tag.split('}', 1)[1]
                    
                    # Extract text from paragraphs and other text elements
                    text_elements = []
                    
                    # Look for common text elements
                    for elem in root.iter():
                        if elem.tag in ['p', 'h', 'span', 'text:p', 'text:h', 'text:span']:
                            if elem.text:
                                text_elements.append(elem.text)
                            # Also get text from nested elements
                            for child in elem:
                                if child.text:
                                    text_elements.append(child.text)
                                if child.tail:
                                    text_elements.append(child.tail)
                    
                    text_content = '\n'.join(text_elements)
                    
                except KeyError:
                    logger.error("content.xml not found in ODT file")
                    return "Could not extract content from ODT file - invalid format"
                except ET.ParseError as e:
                    logger.error(f"Error parsing ODT XML: {e}")
                    return "Could not parse ODT file content"
            
            if not text_content.strip():
                logger.warning("No text extracted from ODT file")
                return "No text content could be extracted from this ODT file."
            
            logger.info(f"✅ Extracted {len(text_content)} characters from ODT")
            return self._clean_text(text_content)
            
        except zipfile.BadZipFile:
            logger.error("Invalid ODT file - not a valid ZIP archive")
            raise ValueError("Invalid ODT file format")
        except Exception as e:
            logger.error(f"Error processing ODT: {str(e)}")
            raise
    
    def _clean_markdown_syntax(self, text: str) -> str:
        """Remove common markdown syntax for plain text fallback."""
        # Remove headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
        
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_content(self, content: str, file_name: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split content into overlapping chunks for vector storage.
        
        Args:
            content: Text content to chunk
            file_name: Original filename
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
        
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not content or not content.strip():
            logger.warning("No content to chunk")
            return []
        
        chunks = []
        content = content.strip()
        
        # If content is smaller than chunk size, return as single chunk
        if len(content) <= chunk_size:
            chunks.append({
                'content': content,
                'metadata': {
                    'file_name': file_name,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'char_count': len(content),
                    'created_at': datetime.now().isoformat()
                }
            })
            return chunks
        
        # Split into overlapping chunks
        start = 0
        chunk_index = 0
        
        while start < len(content):
            # Calculate end position
            end = start + chunk_size
            
            # If this isn't the last chunk, try to find a good break point
            if end < len(content):
                # Look for sentence boundaries within the last 200 characters
                search_start = max(start + chunk_size - 200, start + chunk_size // 2)
                
                # Try to find sentence endings
                for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    punct_pos = content.rfind(punct, search_start, end)
                    if punct_pos != -1:
                        end = punct_pos + len(punct)
                        break
                else:
                    # If no sentence boundary, try paragraph breaks
                    para_pos = content.rfind('\n\n', search_start, end)
                    if para_pos != -1:
                        end = para_pos + 2
                    else:
                        # If no paragraph break, try word boundaries
                        space_pos = content.rfind(' ', search_start, end)
                        if space_pos != -1:
                            end = space_pos + 1
            
            # Extract chunk
            chunk_content = content[start:end].strip()
            
            if chunk_content:  # Only add non-empty chunks
                chunks.append({
                    'content': chunk_content,
                    'metadata': {
                        'file_name': file_name,
                        'chunk_index': chunk_index,
                        'char_start': start,
                        'char_end': end,
                        'char_count': len(chunk_content),
                        'created_at': datetime.now().isoformat()
                    }
                })
                chunk_index += 1
            
            # Move start position for next chunk (with overlap)
            start = end - chunk_overlap
            
            # Prevent infinite loops
            if start >= end:
                start = end
        
        # Update total chunks in metadata
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        logger.info(f"✂️ Split content into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")
        return chunks
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with OCR fallback"""
        try:
            reader = PdfReader(file_path)
            text = ""
            
            # Try normal text extraction first
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # If minimal text extracted, use OCR
            if len(text.strip()) < 100:
                logger.warning("Minimal text extracted, attempting OCR...")
                text = self._extract_text_with_ocr(file_path)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            # Fallback to OCR if normal extraction fails
            try:
                return self._extract_text_with_ocr(file_path)
            except Exception as ocr_error:
                logger.error(f"OCR extraction also failed: {str(ocr_error)}")
                raise
    
    def _extract_text_with_ocr(self, file_path: str) -> str:
        """Extract text from image-based PDF using OCR"""
        try:
            logger.info("🔍 Starting OCR extraction...")
            
            # Convert PDF to images
            images = convert_from_path(file_path, dpi=300)
            logger.info(f"📄 Converted PDF to {len(images)} images")
            
            text = ""
            for i, image in enumerate(images):
                logger.info(f"🔎 Processing page {i+1}/{len(images)} with OCR...")
                
                # Extract text using Tesseract
                page_text = pytesseract.image_to_string(image, lang='eng')
                text += page_text + "\n\n"
                
                logger.info(f"✅ Page {i+1}: Extracted {len(page_text)} characters")
            
            logger.info(f"✅ OCR completed: Total {len(text)} characters extracted")
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {str(e)}")
            raise
        
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF files with OCR fallback."""
        try:
            text_content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Page {page_num + 1} ---\n"
                            text_content += page_text
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            # Check if we got meaningful text (more than just whitespace/metadata)
            if not text_content.strip() or len(text_content.strip()) < 100:
                logger.warning(f"Minimal text extracted ({len(text_content.strip())} chars), attempting OCR...")
                try:
                    text_content = self._extract_text_with_ocr(file_path)
                    logger.info(f"✅ OCR extraction successful: {len(text_content)} characters")
                except Exception as ocr_error:
                    logger.error(f"❌ OCR extraction failed: {str(ocr_error)}")
                    # Return what we have rather than failing completely
                    if text_content.strip():
                        logger.warning("Using minimal extracted text as fallback")
                    else:
                        return "No text content could be extracted from this PDF file. The file may be corrupted or encrypted."
            
            logger.info(f"✅ Extracted {len(text_content)} characters from PDF")
            return self._clean_text(text_content)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def _extract_text_with_ocr(self, file_path: str) -> str:
        """Extract text from image-based PDF using OCR"""
        try:
            logger.info("🔍 Starting OCR extraction...")
            
            # Convert PDF to images with higher DPI for better OCR accuracy
            images = convert_from_path(file_path, dpi=300)
            logger.info(f"📄 Converted PDF to {len(images)} images")
            
            text = ""
            for i, image in enumerate(images):
                logger.info(f"🔎 Processing page {i+1}/{len(images)} with OCR...")
                
                # Extract text using Tesseract with English language
                # You can add more languages if needed: lang='eng+spa'
                page_text = pytesseract.image_to_string(image, lang='eng')
                
                if page_text.strip():
                    text += f"\n--- Page {i+1} (OCR) ---\n"
                    text += page_text + "\n\n"
                    logger.info(f"✅ Page {i+1}: Extracted {len(page_text)} characters")
                else:
                    logger.warning(f"⚠️ Page {i+1}: No text extracted")
            
            if not text.strip():
                logger.warning("❌ OCR completed but no text was extracted")
                return "OCR processing completed but no readable text was found in the PDF."
            
            logger.info(f"✅ OCR completed: Total {len(text)} characters extracted from {len(images)} pages")
            return text.strip()
            
        except ImportError as e:
            logger.error(f"❌ OCR dependencies not installed: {str(e)}")
            raise ValueError(
                "OCR processing requires additional packages. "
                "Install with: pip install pdf2image pytesseract pillow\n"
                "Also ensure Tesseract OCR is installed on your system:\n"
                "- macOS: brew install tesseract\n"
                "- Ubuntu: sudo apt-get install tesseract-ocr\n"
                "- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {str(e)}")
            raise
        
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF files with configurable OCR fallback."""
        try:
            text_content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Page {page_num + 1} ---\n"
                            text_content += page_text
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            # Check if OCR is needed based on configuration threshold
            min_threshold = getattr(self.config, 'OCR_MIN_TEXT_THRESHOLD', 100)
            enable_ocr = getattr(self.config, 'ENABLE_OCR', True)
            
            if enable_ocr and (not text_content.strip() or len(text_content.strip()) < min_threshold):
                logger.warning(
                    f"Minimal text extracted ({len(text_content.strip())} chars, "
                    f"threshold: {min_threshold}), attempting OCR..."
                )
                try:
                    ocr_text = self._extract_text_with_ocr(file_path)
                    if ocr_text and len(ocr_text.strip()) > len(text_content.strip()):
                        logger.info(f"✅ OCR extraction successful: {len(ocr_text)} characters")
                        text_content = ocr_text
                    else:
                        logger.warning("OCR did not improve text extraction, using original")
                except Exception as ocr_error:
                    logger.error(f"❌ OCR extraction failed: {str(ocr_error)}")
                    if not text_content.strip():
                        return "No text content could be extracted from this PDF file. OCR processing failed."
            
            if not text_content.strip():
                return "No text content could be extracted from this PDF file."
            
            logger.info(f"✅ Extracted {len(text_content)} characters from PDF")
            return self._clean_text(text_content)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _extract_text_with_ocr(self, file_path: str) -> str:
        """Extract text from image-based PDF using OCR with configuration."""
        try:
            logger.info("🔍 Starting OCR extraction...")
            
            # Get DPI from config
            dpi = getattr(self.config, 'OCR_DPI', 300)
            languages = getattr(self.config, 'OCR_LANGUAGES', 'eng')
            
            # Convert PDF to images
            logger.info(f"📄 Converting PDF to images (DPI: {dpi})...")
            images = convert_from_path(file_path, dpi=dpi)
            logger.info(f"📄 Converted PDF to {len(images)} images")
            
            text = ""
            successful_pages = 0
            
            for i, image in enumerate(images):
                logger.info(f"🔎 Processing page {i+1}/{len(images)} with OCR (lang: {languages})...")
                
                try:
                    # Extract text using Tesseract
                    page_text = pytesseract.image_to_string(image, lang=languages)
                    
                    if page_text.strip():
                        text += f"\n--- Page {i+1} (OCR) ---\n"
                        text += page_text + "\n\n"
                        successful_pages += 1
                        logger.info(f"✅ Page {i+1}: Extracted {len(page_text)} characters")
                    else:
                        logger.warning(f"⚠️ Page {i+1}: No text extracted")
                except Exception as page_error:
                    logger.error(f"❌ Error processing page {i+1}: {str(page_error)}")
                    continue
            
            if not text.strip():
                logger.warning("❌ OCR completed but no text was extracted from any page")
                return "OCR processing completed but no readable text was found in the PDF."
            
            logger.info(
                f"✅ OCR completed: Total {len(text)} characters extracted "
                f"from {successful_pages}/{len(images)} pages"
            )
            return text.strip()
            
        except ImportError as e:
            logger.error(f"❌ OCR dependencies not installed: {str(e)}")
            raise ValueError(
                "OCR processing requires additional packages:\n"
                "  pip install pdf2image pytesseract pillow PyPDF2\n\n"
                "System dependencies:\n"
                "  macOS: brew install tesseract poppler\n"
                "  Ubuntu: sudo apt-get install tesseract-ocr poppler-utils\n"
                "  Windows: See https://github.com/UB-Mannheim/tesseract/wiki"
            )
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {str(e)}")
            raise
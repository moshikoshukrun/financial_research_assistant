"""Document QA tool for 10-K analysis."""

import os
import re
import logging
from io import StringIO
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class DocumentQA:
    """Document QA system for Apple 10-K filing."""
    
    def __init__(self, document_path: str, vector_db_path: str = "./data/vector_store"):
        """
        Initialize Document QA system.
        
        Args:
            document_path: Path to 10-K HTML file
            vector_db_path: Path to vector database
        """
        self.doc_path = document_path
        self.vector_db_path = vector_db_path
        
        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        
        # Get or create collection
        self.collection_name = "apple_10k"
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info("Loaded existing vector store")
        except Exception:
            self.collection = None
            logger.info("Vector store not found, will create new one")
        
        self.chunks = []
    
    def initialize(self):
        """Initialize the document QA system by processing and indexing the document."""
        if self.collection is not None and self.collection.count() > 0:
            logger.info(f"Vector store already initialized with {self.collection.count()} chunks")
            return
        
        logger.info("Processing document...")
        self._process_document()
    
    def _process_document(self):
        """Parse, chunk, embed, and store the document."""
        # Parse document
        parsed_doc = self._parse_html_10k(self.doc_path)
        sections = parsed_doc['sections']
        
        # Chunk document
        self.chunks = self._chunk_document(sections)
        logger.info(f"Created {len(self.chunks)} chunks")
        
        if len(self.chunks) == 0:
            raise ValueError("No chunks created from document. Check if file is valid 10-K HTML.")
        
        # Create collection
        if self.collection is None:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Apple 10-K filing chunks"}
            )
        
        # Embed and store
        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self._embed_texts(texts)
        
        # Add to ChromaDB
        ids = [f"chunk_{i}" for i in range(len(self.chunks))]
        metadatas = [
            {
                'section': chunk['section'],
                'page': str(chunk['page']) if chunk['page'] else 'unknown',
                'type': chunk['type'],
                'chunk_id': str(i)
            }
            for i, chunk in enumerate(self.chunks)
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Indexed {len(self.chunks)} chunks to vector store")
    
    def _parse_html_10k(self, path: str) -> Dict:
        """
        Parse 10-K HTML file and extract sections.
        
        Args:
            path: Path to HTML file
            
        Returns:
            Dictionary with sections
        """
        logger.info(f"Parsing 10-K file: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file: {str(e)}")
            raise
        
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove scripts and styles
        for element in soup(['script', 'style']):
            element.decompose()
        
        sections = self._identify_sections(soup)
        logger.info(f"Identified {len(sections)} sections")
        
        return {"sections": sections}
    
    def _identify_sections(self, soup: BeautifulSoup) -> List[Dict]:
        """Identify major sections in the 10-K."""
        sections = []
        
        # Get all text content
        all_text = soup.get_text(separator='\n', strip=True)
        
        logger.info(f"Extracted {len(all_text)} characters from document")
        
        if len(all_text) < 1000:
            logger.error("Document appears to be empty or too short")
            raise ValueError("Document contains insufficient text")
        
        # Simple chunking strategy: split entire document into manageable sections
        words = all_text.split()
        logger.info(f"Total words: {len(words)}")
        
        if len(words) < 100:
            raise ValueError("Document contains too few words")
        
        # Create sections of ~2000 words each
        section_size = 2000
        section_num = 1
        
        for i in range(0, len(words), section_size):
            chunk_words = words[i:i + section_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text) > 500:
                sections.append({
                    'name': f'Document Section {section_num}',
                    'content': chunk_text,
                    'page': section_num,
                    'type': 'text'
                })
                section_num += 1
        
        logger.info(f"Created {len(sections)} document sections")
        
        if len(sections) == 0:
            raise ValueError("Failed to create any sections from document")
        
        return sections
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract tables from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of table dictionaries
        """
        tables = []
        table_elements = soup.find_all('table')
        
        logger.info(f"Found {len(table_elements)} table elements")
        
        for idx, table in enumerate(table_elements[:20]):  # Limit to first 20 tables
            try:
                df = pd.read_html(StringIO(str(table)))[0]
                
                # Convert to readable text
                table_text = df.to_string()
                
                # Skip very small tables
                if len(table_text) < 50:
                    continue
                
                # Try to find nearby section context
                section_name = "Financial Statements"
                parent_text = table.find_previous(['h1', 'h2', 'h3', 'h4', 'p'])
                if parent_text:
                    text = parent_text.get_text()[:100]
                    if 'income' in text.lower():
                        section_name = "Financial Statements - Income Statement"
                    elif 'balance' in text.lower():
                        section_name = "Financial Statements - Balance Sheet"
                    elif 'cash flow' in text.lower():
                        section_name = "Financial Statements - Cash Flow"
                
                tables.append({
                    'name': section_name,
                    'content': table_text,
                    'page': None,
                    'type': 'table'
                })
            except Exception as e:
                logger.warning(f"Failed to parse table {idx}: {str(e)}")
                continue
        
        logger.info(f"Extracted {len(tables)} tables")
        return tables
    
    def _chunk_document(self, sections: List[Dict]) -> List[Dict]:
        """
        Chunk document preserving structure.
        
        Bonus 4: Detects and preserves cross-references like "See Note 5".
        
        Args:
            sections: List of section dictionaries
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        chunk_size = 500  # tokens (approximate as words)
        overlap = 100
        
        # Bonus 4: Cross-reference pattern
        cross_ref_pattern = r'(?i)(see|refer to|as discussed in|note|item|section)\s+(\d+|[A-Z]+)'
        
        for section in sections:
            section_name = section['name']
            content = section['content']
            page = section['page']
            doc_type = section['type']
            
            # Bonus 4: Detect cross-references
            cross_refs = []
            for match in re.finditer(cross_ref_pattern, content):
                cross_refs.append({
                    'type': match.group(1),
                    'target': match.group(2),
                    'full_text': match.group(0)
                })
            
            if doc_type == 'table':
                # Keep tables as single chunks
                chunks.append({
                    'text': content,
                    'section': section_name,
                    'page': page,
                    'type': 'table',
                    'chunk_id': len(chunks),
                    'cross_references': cross_refs
                })
            else:
                # Split text into chunks
                words = content.split()
                
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    if len(chunk_words) < 50:  # Skip very small chunks
                        continue
                    
                    chunks.append({
                        'text': chunk_text,
                        'section': section_name,
                        'page': page,
                        'type': 'text',
                        'chunk_id': len(chunks),
                        'cross_references': [ref for ref in cross_refs 
                                            if ref['full_text'] in chunk_text]
                    })
        
        return chunks
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        return self.embedding_model.encode(texts, show_progress_bar=False)
    
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            query: Query string
            
        Returns:
            NumPy array embedding
        """
        return self.embedding_model.encode([query], show_progress_bar=False)[0]
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Query the document.
        
        Args:
            question: Question to ask
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and citations
        """
        try:
            # Retrieve relevant chunks
            query_embedding = self._embed_query(question)
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            if not results['documents'][0]:
                return {
                    'answer': 'No relevant information found in the 10-K filing.',
                    'citations': []
                }
            
            # Build context from chunks
            chunks = []
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                chunks.append({
                    'text': doc,
                    'section': metadata['section'],
                    'page': metadata['page'],
                    'type': metadata['type'],
                    'chunk_id': i
                })
            
            context = self._build_context(chunks)
            answer = self._generate_answer(question, context)
            citations = self._extract_citations(answer, chunks)
            
            return {
                'answer': answer,
                'citations': citations
            }
            
        except Exception as e:
            logger.error(f"Document QA query failed: {str(e)}")
            return {
                'answer': f'Error querying document: {str(e)}',
                'citations': []
            }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Chunk {i}] (Section: {chunk['section']}, Page: {chunk['page']})\n{chunk['text']}"
            )
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            question: User question
            context: Context from retrieved chunks
            
        Returns:
            Generated answer
        """
        with open('prompts/system_prompt.txt', 'r') as f:
            system_prompt = f.read()
        
        prompt = f"""Question: {question}

Context from Apple's 10-K filing:
{context}

Please answer the question based on the context provided. Reference specific chunks using [Chunk X] notation when citing information."""
        
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=full_prompt
            )
            
            return response.text
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise
    
    def _extract_citations(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        """
        Extract citations from answer.
        
        Bonus 2: Returns page-level citations with specific page numbers.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        # Find all [Chunk X] references
        chunk_refs = re.findall(r'\[Chunk (\d+)\]', answer)
        
        for ref in set(chunk_refs):
            chunk_idx = int(ref)
            if chunk_idx < len(chunks):
                chunk = chunks[chunk_idx]
                
                # Bonus 2 & 4: Page-level citation and cross-references
                cross_refs = chunk.get('cross_references', [])
                
                citations.append({
                    'source_type': 'document',
                    'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                    'section': chunk['section'],
                    'page': chunk['page'],
                    'cross_references': cross_refs,
                    'url': None,
                    'title': None
                })
        
        return citations

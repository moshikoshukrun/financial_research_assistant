"""Tavily search tool for web information."""

import os
import logging
from typing import Dict, List
import requests

logger = logging.getLogger(__name__)


class TavilySearch:
    """Tavily search API wrapper."""
    
    def __init__(self):
        """Initialize Tavily search client."""
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        self.base_url = "https://api.tavily.com/search"
        logger.info("Initialized Tavily search client")
    
    def search(self, query: str, max_results: int = 5) -> Dict:
        """
        Search the web using Tavily.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            logger.info(f"Searching Tavily for: {query}")
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic"
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 429:
                logger.warning("Tavily rate limit exceeded")
                return {
                    'answer': 'Rate limit exceeded for web search. Please try again later.',
                    'sources': []
                }
            
            response.raise_for_status()
            data = response.json()
            
            # Format results
            formatted = self._format_results(data)
            logger.info(f"Found {len(formatted['sources'])} results")
            
            return formatted
            
        except requests.exceptions.Timeout:
            logger.error("Tavily request timed out")
            return {
                'answer': 'Web search timed out. Please try again.',
                'sources': []
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Tavily search failed: {str(e)}")
            return {
                'answer': f'Web search error: {str(e)}',
                'sources': []
            }
        except Exception as e:
            logger.error(f"Unexpected error in Tavily search: {str(e)}")
            return {
                'answer': f'Error performing web search: {str(e)}',
                'sources': []
            }
    
    def _format_results(self, raw_results: Dict) -> Dict:
        """
        Format Tavily API results.
        
        Args:
            raw_results: Raw API response
            
        Returns:
            Formatted results with citations
        """
        sources = []
        
        results = raw_results.get('results', [])
        for result in results:
            sources.append({
                'source_type': 'web',
                'text': result.get('content', '')[:200] + '...',
                'section': None,
                'page': None,
                'url': result.get('url'),
                'title': result.get('title')
            })
        
        # Build summary answer from results
        answer_parts = []
        for i, result in enumerate(results[:3]):  # Top 3 results
            content = result.get('content', '')
            if content:
                answer_parts.append(f"[Source {i+1}]: {content[:300]}")
        
        answer = '\n\n'.join(answer_parts) if answer_parts else 'No results found.'
        
        return {
            'answer': answer,
            'sources': sources
        }
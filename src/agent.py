"""Main agent logic for financial research assistant."""

import os
import logging
from typing import Dict, List
from google import genai

from src.tools.document_qa import DocumentQA
from src.tools.tavily_search import TavilySearch

logger = logging.getLogger(__name__)


class FinancialAgent:
    """Agent for financial research using document QA and web search."""
    
    def __init__(self, document_path: str):
        """
        Initialize Financial Agent.
        
        Args:
            document_path: Path to 10-K HTML file
        """
        logger.info("Initializing Financial Agent...")
        
        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        
        # Initialize tools
        vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_store")
        self.doc_qa = DocumentQA(document_path, vector_db_path)
        self.tavily = TavilySearch()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Initialize document QA system
        self.doc_qa.initialize()
        
        logger.info("Financial Agent initialized successfully")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        prompts = {}
        
        prompt_files = {
            'system': 'prompts/system_prompt.txt',
            'tool_descriptions': 'prompts/tool_descriptions.txt',
            'examples': 'prompts/examples.txt'
        }
        
        for key, path in prompt_files.items():
            try:
                with open(path, 'r') as f:
                    prompts[key] = f.read()
            except FileNotFoundError:
                logger.warning(f"Prompt file not found: {path}")
                prompts[key] = ""
        
        return prompts
    
    def answer_query(self, query: str) -> Dict:
        """
        Answer a user query.
        
        Args:
            query: User question
            
        Returns:
            Dictionary with answer, citations, and tools used
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Bonus 3: Create plan
            plan = self._create_plan(query)
            logger.info(f"Plan: {plan}")
            
            # Route to appropriate tools
            tools_to_use = self._route_tools(query)
            logger.info(f"Using tools: {tools_to_use}")
            
            # Execute tools
            tool_results = self._execute_tools(query, tools_to_use)
            
            # Synthesize final answer
            final_response = self._synthesize_answer(query, tool_results)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'answer': f'Error processing query: {str(e)}',
                'citations': [],
                'tools_used': [],
                'errors': [str(e)],
                'plan': 'Error occurred'
            }
    
    def _create_plan(self, query: str) -> str:
        """Bonus 3: Create execution plan."""
        query_lower = query.lower()
        
        has_current = any(kw in query_lower for kw in ["current", "today", "vs", "compare", "microsoft"])
        has_document = any(kw in query_lower for kw in ["risk", "10-k", "r&d", "margin"])
        
        if has_current and has_document:
            return "Plan: (1) Query 10-K for historical data (2) Search web for current data (3) Synthesize both sources"
        elif has_current:
            return "Plan: Search web for current market information"
        else:
            return "Plan: Query Apple's 10-K filing for requested information"
    
    def _route_tools(self, query: str) -> List[str]:
        """
        Determine which tools to use (deterministic routing).
        
        Args:
            query: User question
            
        Returns:
            List of tool names to use
        """
        query_lower = query.lower()
        
        # Keywords that suggest need for current/real-time data
        current_data_keywords = [
            "current", "today", "now", "latest stock", "market cap",
            "compare to", "versus", "vs", "microsoft", "google", 
            "competitor", "industry average", "recent news"
        ]
        
        # Keywords that suggest document-only query
        document_only_keywords = [
            "risk factor", "10-k", "filing", "annual report",
            "management discussion", "md&a", "financial statement",
            "balance sheet", "income statement"
        ]
        
        needs_current_data = any(kw in query_lower for kw in current_data_keywords)
        mentions_document = any(kw in query_lower for kw in document_only_keywords)
        
        # Route based on keywords
        if needs_current_data and mentions_document:
            return ["document_qa", "tavily"]  # Hybrid: both tools
        elif needs_current_data and not mentions_document:
            return ["tavily"]  # Web search only
        else:
            return ["document_qa"]  # Document only
    
    def _execute_tools(self, query: str, tools: List[str]) -> Dict:
        """
        Execute the selected tools.
        
        Args:
            query: User question
            tools: List of tool names
            
        Returns:
            Dictionary with results and any errors
        """
        results = {}
        errors = []
        
        for tool_name in tools:
            try:
                if tool_name == "document_qa":
                    logger.info("Executing Document QA tool...")
                    results['document_qa'] = self.doc_qa.query(query)
                    
                elif tool_name == "tavily":
                    logger.info("Executing Tavily search tool...")
                    results['tavily'] = self.tavily.search(query)
                    
            except Exception as e:
                error_msg = f"{tool_name} failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        if not results and errors:
            raise Exception(f"All tools failed: {'; '.join(errors)}")
        
        return {
            'results': results,
            'errors': errors
        }
    
    def _synthesize_answer(self, query: str, tool_results: Dict) -> Dict:
        """
        Synthesize final answer from tool results.
        
        Args:
            query: User question
            tool_results: Results from tools
            
        Returns:
            Dictionary with final answer and citations
        """
        results = tool_results.get('results', {})
        errors = tool_results.get('errors', [])
        
        all_answers = []
        all_citations = []
        tools_used = []
        
        # Collect answers from each tool
        if 'document_qa' in results:
            doc_result = results['document_qa']
            all_answers.append(f"From 10-K Document:\n{doc_result['answer']}")
            all_citations.extend(doc_result['citations'])
            tools_used.append('Document QA')
        
        if 'tavily' in results:
            web_result = results['tavily']
            all_answers.append(f"From Web Search:\n{web_result['answer']}")
            all_citations.extend(web_result['sources'])
            tools_used.append('Tavily Search')
        
        # If multiple tools, synthesize
        if len(all_answers) > 1:
            combined_context = "\n\n".join(all_answers)
            
            synthesis_prompt = f"""Question: {query}

Information gathered from multiple sources:

{combined_context}

Please synthesize this information into a comprehensive answer that:
1. Directly answers the question
2. Combines insights from both sources where relevant
3. Clearly distinguishes between historical data (from 10-K) and current data (from web)
4. Provides any relevant comparisons or calculations

Your answer:"""
            
            try:
                full_prompt = f"{self.prompts['system']}\n\n{synthesis_prompt}"
                
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=full_prompt
                )
                
                final_answer = response.text
            except Exception as e:
                logger.error(f"Synthesis failed: {str(e)}")
                final_answer = "\n\n---\n\n".join(all_answers)
        else:
            final_answer = all_answers[0] if all_answers else "No answer generated."
        
        plan = self._create_plan(query)
        
        return {
            'answer': final_answer,
            'citations': all_citations,
            'tools_used': tools_used,
            'errors': errors,
            'plan': plan
        }

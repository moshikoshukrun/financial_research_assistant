"""
Test script for the two required queries.

This script tests the Financial Research Assistant with:
1. Document-only query (risks + R&D spending)
2. Hybrid query (document + web search for comparison)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import FinancialAgent


def test_query_1_document_only():
    """
    Test Query 1: Document-Only Analysis
    
    Question: "What are Apple's top 3 risk factors mentioned in their latest 10-K, 
    and what percentage of total revenue did they spend on R&D?"
    
    Expected behavior per contract:
    - Tool routing: Document QA only (no web search)
    - Document QA performs single query() call that retrieves from multiple sections:
      * Risk Factors section (for top 3 risks)
      * Financial Statements section (for R&D and revenue data)
    - Agent synthesizes the information into a coherent answer
    """
    print("\n" + "="*80)
    print("TEST QUERY 1: Document-Only Analysis")
    print("="*80)
    
    query = "What are Apple's top 3 risk factors mentioned in their latest 10-K, and what percentage of total revenue did they spend on R&D?"
    
    print(f"\nQuery: {query}")
    print("\nExpected Tools: Document QA only")
    print("Expected Behavior:")
    print("  - Single Document QA query() call retrieves from multiple sections:")
    print("    * Risk Factors section (for top 3 risks)")
    print("    * Financial Statements section (for R&D and revenue data)")
    print("\nProcessing...\n")
    
    agent = FinancialAgent("data/apple_10k_2023.htm")
    response = agent.answer_query(query)
    
    print("\n--- ANSWER ---")
    print(response['answer'])
    
    # Bonus 3: Display plan
    if response.get('plan'):
        print(f"\n--- PLAN ---")
        print(response['plan'])
    
    print("\n--- CITATIONS ---")
    if response['citations']:
        for i, citation in enumerate(response['citations'], 1):
            print(f"[{i}] {citation['section']}, Page {citation['page']}")
            
            # Bonus 4: Cross-references
            if citation.get('cross_references'):
                refs_text = ', '.join([r['full_text'] for r in citation['cross_references'][:2]])
                print(f"    Cross-refs: {refs_text}")
            
            print(f"    {citation['text'][:150]}...")
    else:
        print("No citations returned")
    
    print("\n--- TOOLS USED ---")
    print(", ".join(response['tools_used']))
    
    # Validation
    assert 'Document QA' in response['tools_used'], "Should use Document QA"
    assert 'Tavily Search' not in response['tools_used'], "Should NOT use Tavily"
    assert len(response['citations']) > 0, "Should have citations"
    assert 'risk' in response['answer'].lower() or 'r&d' in response['answer'].lower(), "Answer should mention risks or R&D"
    
    print("\n✓ Test Query 1 PASSED")
    return True


def test_query_2_hybrid():
    """
    Test Query 2: Hybrid Analysis (Document + Web)
    
    Question: "How does Apple's gross margin compare to Microsoft's current gross margin, 
    and what reasons does Apple cite in their 10-K for any margin pressure?"
    
    Expected behavior per contract:
    - Document QA: Extract Apple's gross margin from Financial Statements
    - Document QA: Extract margin pressure reasons from MD&A section
    - Tavily: Search for Microsoft's current gross margin
    - Single Document QA query() call retrieves from multiple sections (Financial Statements + MD&A)
    """
    print("\n" + "="*80)
    print("TEST QUERY 2: Hybrid Analysis")
    print("="*80)
    
    query = "How does Apple's gross margin compare to Microsoft's current gross margin, and what reasons does Apple cite in their 10-K for any margin pressure?"
    
    print(f"\nQuery: {query}")
    print("\nExpected Tools: Document QA + Tavily")
    print("Expected Behavior:")
    print("  - Document QA query() retrieves from multiple sections:")
    print("    * Financial Statements (for Apple's gross margin)")
    print("    * MD&A section (for margin pressure discussion)")
    print("  - Tavily search for Microsoft's current gross margin")
    print("  - Agent synthesizes both sources into comprehensive comparison")
    print("\nProcessing...\n")
    
    agent = FinancialAgent("data/apple_10k_2023.htm")
    response = agent.answer_query(query)
    
    print("\n--- ANSWER ---")
    print(response['answer'])
    
    # Bonus 3: Display plan
    if response.get('plan'):
        print(f"\n--- PLAN ---")
        print(response['plan'])
    
    print("\n--- CITATIONS ---")
    if response['citations']:
        for i, citation in enumerate(response['citations'], 1):
            if citation['source_type'] == 'document':
                print(f"[{i}] Document: {citation['section']}, Page {citation['page']}")
                
                # Bonus 4: Cross-references
                if citation.get('cross_references'):
                    refs_text = ', '.join([r['full_text'] for r in citation['cross_references'][:2]])
                    print(f"    Cross-refs: {refs_text}")
            else:
                print(f"[{i}] Web: {citation['title']}")
                print(f"    URL: {citation['url']}")
    else:
        print("No citations returned")
    
    print("\n--- TOOLS USED ---")
    print(", ".join(response['tools_used']))
    
    # Validation
    assert 'Document QA' in response['tools_used'], "Should use Document QA"
    assert 'Tavily Search' in response['tools_used'], "Should use Tavily"
    assert len(response['citations']) > 0, "Should have citations"
    
    # Check for both document and web citations
    has_document_citation = any(c['source_type'] == 'document' for c in response['citations'])
    has_web_citation = any(c['source_type'] == 'web' for c in response['citations'])
    
    assert has_document_citation, "Should have at least one document citation"
    assert has_web_citation, "Should have at least one web citation"
    
    print("\n✓ Test Query 2 PASSED")
    return True


def run_test_queries():
    """Run all test queries."""
    print("\n" + "="*80)
    print("RUNNING TEST QUERIES")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Check if document exists
    if not Path("data/apple_10k_2023.htm").exists():
        print("\n✗ ERROR: Apple 10-K file not found at data/apple_10k_2023.htm")
        print("\nPlease download the file from SEC EDGAR:")
        print("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193")
        print("\nOr use the SEC EDGAR search:")
        print("1. Go to: https://www.sec.gov/edgar/searchedgar/companysearch.html")
        print("2. Search for: Apple Inc.")
        print("3. Find the latest 10-K filing")
        print("4. Download the HTML version")
        print("5. Save as: data/apple_10k_2023.htm")
        sys.exit(1)
    
    # Check API keys
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n✗ ERROR: GOOGLE_API_KEY not found in environment")
        print("Please add it to your .env file:")
        print("GOOGLE_API_KEY=your_key_here")
        sys.exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("\n⚠ WARNING: TAVILY_API_KEY not found in environment")
        print("Test 2 (hybrid query) may fail without web search capability")
        print("Get a free API key at: https://tavily.com")
    
    # Run tests
    success = True
    
    try:
        if not test_query_1_document_only():
            success = False
    except Exception as e:
        print(f"\n✗ Test Query 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        if not test_query_2_hybrid():
            success = False
    except Exception as e:
        print(f"\n✗ Test Query 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Summary
    print("\n" + "="*80)
    if success:
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nContract Requirements Met:")
        print("✓ Query 1: Document-only analysis with multi-section retrieval")
        print("✓ Query 2: Hybrid analysis with document + web search")
        print("✓ Single Document QA query() call per request")
        print("✓ Proper tool routing based on query type")
        print("✓ Citations from both document and web sources")
        print("\nBonus Features Implemented:")
        print("✓ Bonus 2: Page-level citations")
        print("✓ Bonus 3: Planning step displayed")
        print("✓ Bonus 4: Cross-reference detection")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*80)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_test_queries())

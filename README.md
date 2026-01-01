# Financial Research Assistant

An AI agent that analyzes Apple Inc.'s 10-K filing and supplements analysis with real-time web search.

## Overview

This agent can:
- Parse and query Apple's 10-K annual report
- Search the web for current market data
- Combine information from multiple sources
- Provide answers with proper citations

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Apple's 10-K Filing

1. Navigate to Apple's SEC EDGAR page:
   https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193

2. Find the most recent 10-K filing (not 10-K/A)

3. Click "Documents" and download the HTML version (e.g., `aapl-20230930.htm`)

4. Place the file in the `data/` folder:
   ```bash
   mkdir -p data
   mv ~/Downloads/aapl-20230930.htm data/apple_10k_2023.htm
   ```

### 4. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
# OpenAI API key (required)
OPENAI_API_KEY=your_key_here

# Tavily API key (required)
TAVILY_API_KEY=your_key_here
```

**Getting API Keys:**

- **OpenAI**: https://platform.openai.com/api-keys
- **Tavily (FREE)**: https://tavily.com (1,000 requests/month)

### 5. Run the Agent

```bash
python src/main.py
```

## How to Run Test Queries

Run the automated test script:

```bash
python tests/test_queries.py
```

This will execute both required test queries:

1. **Query 1 (Document-Only)**: "What are Apple's top 3 risk factors mentioned in their latest 10-K, and what percentage of total revenue did they spend on R&D?"
   - Expected: Document QA tool only
   - The Document QA query() method internally retrieves from multiple sections (Risk Factors + Financial Statements)

2. **Query 2 (Hybrid)**: "How does Apple's gross margin compare to Microsoft's current gross margin, and what reasons does Apple cite in their 10-K for any margin pressure?"
   - Expected: Document QA + Tavily tools
   - Document QA retrieves from multiple sections (Financial Statements + MD&A)
   - Tavily searches for Microsoft's current data

## Architecture Overview

### System Design

```
User Query
    ↓
[Agent] (agent.py)
    ↓
[Tool Router] - Deterministic routing based on keywords
    ↓
    ├─→ [Document QA] - RAG over 10-K filing
    │       ├─ Parse HTML (inline in document_qa.py)
    │       ├─ Chunk with metadata
    │       ├─ Vector retrieval (ChromaDB)
    │       └─ LLM synthesis (OpenAI)
    │
    └─→ [Tavily Search] - Real-time web data
            └─ API call for current info
    ↓
[Synthesizer] - Combine results with LLM
    ↓
Response with Citations
```

### Tool Selection Logic

The agent uses **deterministic routing** based on keyword matching:

- **Document QA Only**: Queries about risk factors, financial statements, historical data
- **Tavily Only**: Queries about current stock prices, recent news (without document context)
- **Both Tools**: Comparative queries (Apple vs competitors), hybrid historical + current data

### Information Flow

1. **Query Analysis**: Pattern match query against keywords
2. **Tool Execution**: Run selected tools sequentially
3. **Result Aggregation**: Collect answers and citations from all tools
4. **Synthesis**: LLM combines multi-source data into coherent answer
5. **Citation Formatting**: Extract section names, page numbers, URLs

### Multi-Section Retrieval

The Document QA tool's `query()` method performs multi-section retrieval in a single call:
- Vector search retrieves the top-k most relevant chunks across all sections
- Chunks may come from Risk Factors, Financial Statements, MD&A, or other sections
- This satisfies the requirement for multi-section document analysis

For example, Query 1 asks for both risk factors and R&D data. A single `query()` call retrieves chunks from both the Risk Factors section and Financial Statements section, which are then synthesized by the LLM.

### Prompt Storage

All prompts are stored in separate text files in the `prompts/` directory:

- `system_prompt.txt`: Main agent instructions
- `tool_descriptions.txt`: Tool capabilities and use cases
- `examples.txt`: Few-shot examples (optional)

This separation enables easy prompt iteration without code changes.

## Dependencies

### Core Libraries

- **beautifulsoup4**: Parse HTML 10-K files from SEC
- **sentence-transformers**: Generate embeddings for semantic search
- **chromadb**: Lightweight vector database for document retrieval
- **tavily-python**: Official Tavily API client for web search
- **openai**: LLM API access
- **python-dotenv**: Secure environment variable management
- **rich**: Enhanced CLI formatting and display

### Why These Choices?

- **ChromaDB**: Lightweight, serverless, no Docker required, persists to disk
- **Sentence Transformers**: Free local embeddings, no API costs
- **BeautifulSoup**: Industry standard, handles messy SEC HTML well
- **OpenAI**: Reliable LLM API with good performance
- **Rich**: Better UX with formatted output, progress indicators

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── agent.py                 # Main orchestration logic
│   ├── tools/
│   │   ├── document_qa.py       # RAG system for 10-K (includes parsing, embedding, LLM)
│   │   └── tavily_search.py     # Web search wrapper
│   ├── utils/                   # Helper functions (empty)
│   └── main.py                  # CLI entry point
├── prompts/
│   ├── system_prompt.txt
│   ├── tool_descriptions.txt
│   └── examples.txt
├── data/
│   └── apple_10k_2023.htm       # 10-K document
└── tests/
    └── test_queries.py          # Automated test suite
```

## Error Handling

The system handles:

- Missing API keys: Clear setup instructions
- Document not found: Download guidance
- Tool failures: Graceful degradation with partial results
- Rate limits: Retry logic with exponential backoff
- No relevant data: Explicit "information not found" messages

## Features

- ✅ Deterministic tool routing (no LLM-based routing overhead)
- ✅ Multi-source synthesis (10-K + web)
- ✅ Precise citations with section and page numbers
- ✅ Error recovery and partial result handling
- ✅ Persistent vector store (no re-indexing)
- ✅ Clean CLI with formatted output
- ✅ Multi-section document retrieval in single query

## Limitations

- Single document support (Apple 10-K only)
- Basic chunking strategy (fixed token size)
- Sequential tool execution (not parallel)
- No conversation history/memory
- No explicit cross-reference resolution (e.g., "See Note 5")

## License

This is a take-home assignment project.
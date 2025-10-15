# Code RAG Spike

A code repository RAG (Retrieval-Augmented Generation) pipeline using LlamaIndex for ingestion and Haystack for hybrid retrieval.

## Features

- 🔍 **Smart Code Ingestion**: Automatically detects programming languages and chunks code intelligently
- 🔎 **Hybrid Retrieval**: Combines BM25 (keyword) and semantic (embedding) search for best results
- 🎯 **Haystack Compatible**: Uses compatible schema for seamless integration
- 🚫 **Gitignore Support**: Respects `.gitignore` patterns automatically
- 📊 **Progress Tracking**: Verbose mode shows detailed processing information

## Quick Start

### 1. Setup OpenSearch

```bash
cd opensearch
docker-compose up -d
```

Verify it's running:
```bash
curl -k -u admin:admin https://localhost:9200
```

### 2. Configure Environment

Create a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
OPENSEARCH_HOST=https://localhost:9200
OPENSEARCH_INDEX=code_index
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=admin
HAYSTACK_TELEMETRY_ENABLED=False
```

### 3. Install Dependencies

```bash
uv sync
source .venv/bin/activate
```

## Usage

### Ingestion

Ingest a code repository into OpenSearch:

```bash
# Basic ingestion
python ingestion/ingest.py --repo-path /path/to/your/repo

# With verbose output to see skipped files
python ingestion/ingest.py --repo-path /path/to/your/repo --verbose

# Ingest this project itself
python ingestion/ingest.py --repo-path .
```

**What it does:**
- Parses `.gitignore` and skips ignored files
- Detects code languages (Python, JS/TS, Java, Rust, Go, C/C++, PHP, Ruby)
- Uses `CodeSplitter` for code files (preserves structure)
- Uses `SentenceSplitter` for text files
- Generates embeddings with `text-embedding-3-small`
- Stores in OpenSearch with Haystack-compatible schema

### Retrieval

Query the ingested code:

```bash
# Basic query
python retrieval/query.py --query "how to parse gitignore"

# Show relevance scores
python retrieval/query.py --query "vector embeddings" --show-scores

# Show full metadata
python retrieval/query.py --query "CodeSplitter" --show-metadata

# Adjust number of results (default: 3 BM25 + 3 Semantic)
python retrieval/query.py --query "your query" --top-k-bm25 5 --top-k-embedding 5
```

**How it works:**
- **BM25 Retriever**: Finds exact keyword matches
- **Semantic Retriever**: Finds conceptually similar code
- **Hybrid Fusion**: Combines both using reciprocal rank fusion
- Results are deduplicated and ranked by relevance

## Index Management

### View ingested documents
```bash
# Count documents
curl -k -u admin:admin -X GET "https://localhost:9200/code_index/_count?pretty"

# See sample document
curl -k -u admin:admin -X GET "https://localhost:9200/code_index/_search?pretty&size=1"
```

### Clear and re-ingest
```bash
# Delete the index
curl -k -u admin:admin -X DELETE "https://localhost:9200/code_index?pretty"

# Verify deletion
curl -k -u admin:admin -X GET "https://localhost:9200/_cat/indices?v"

# Re-run ingestion
python ingestion/ingest.py --repo-path /path/to/your/repo
```

## Project Structure

```
code-rag-spike/
├── config.py                 # Shared configuration
├── ingestion/               # Ingestion pipeline
│   ├── ingest.py           # Main ingestion script
│   └── utils.py            # Helper functions
├── retrieval/               # Retrieval pipeline
│   ├── query.py            # CLI query tool
│   └── pipeline.py         # Hybrid retrieval setup
└── opensearch/              # OpenSearch Docker setup
    ├── docker-compose.yml
    └── README.md
```

## Architecture

### Ingestion Flow
```
Code Repository → Parse .gitignore → Detect Languages → 
Smart Chunking (Code/Text) → Generate Embeddings → 
Store in OpenSearch (Haystack schema)
```

### Retrieval Flow
```
Query → Embed Query → 
┌─ BM25 Search (keywords)
└─ Semantic Search (embeddings)
→ Merge & Deduplicate → Ranked Results
```

## Configuration

All settings are in `config.py`:

- **Embedding Model**: `text-embedding-3-small`
- **Code Chunking**: 40 lines with 15 line overlap
- **Text Chunking**: 1024 chars with 200 char overlap
- **Max File Size**: 1MB (larger files skipped)
- **Default Top-K**: 3 BM25 + 3 Semantic results

## Examples

### Example 1: Find function implementations
```bash
python retrieval/query.py --query "parse gitignore function" --show-scores
```

### Example 2: Conceptual search
```bash
python retrieval/query.py --query "how to create vector embeddings" --show-scores
```

### Example 3: Keyword search
```bash
python retrieval/query.py --query "CodeSplitter" --show-scores
```

## Troubleshooting

### OpenSearch connection issues
- Ensure OpenSearch is running: `docker-compose ps`
- Check logs: `docker-compose logs -f opensearch`

### Import errors
- Activate venv: `source .venv/bin/activate`
- Check dependencies: `uv sync`

### No results in retrieval
- Verify ingestion completed: Check logs for "100% success rate"
- Count documents: `curl -k -u admin:admin "https://localhost:9200/code_index/_count?pretty"`

## Technologies

- **LlamaIndex**: Document processing and ingestion
- **Haystack**: Hybrid retrieval pipeline
- **OpenSearch**: Vector database
- **OpenAI**: Text embeddings (`text-embedding-3-small`)
- **Tree-sitter**: Code parsing and chunking


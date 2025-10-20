"""
Shared configuration for ingestion and retrieval pipelines.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenSearch Configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "https://localhost:9200")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "code_index")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Traceloop Configuration
TRACEL_API_KEY = os.getenv("TRACEL_API_KEY")

# Embedding Model
EMBEDDING_MODEL = "text-embedding-3-small"

# Haystack-compatible field mappings
# Note: LlamaIndex stores as "metadata", so we use that for Haystack retrieval
HAYSTACK_FIELD_MAPPING = {
    "embedding_field": "embedding",
    "text_field": "content",
    "content_field": "content",  # For Haystack DocumentStore
    "metadata_field": "metadata",  # Match what LlamaIndex actually stores
}

# Code Splitter Settings (for ingestion)
CODE_SPLITTER_CONFIG = {
    "chunk_lines": 40,
    "chunk_lines_overlap": 15,
    "max_chars": 1500,
}

# Text Splitter Settings (for ingestion)
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1024,
    "chunk_overlap": 200,
}

# File Processing Settings (for ingestion)
MAX_FILE_SIZE_MB = 1  # Skip files larger than this

# Retrieval Settings
DEFAULT_TOP_K_BM25 = 3
DEFAULT_TOP_K_EMBEDDING = 3
DEFAULT_JOIN_MODE = "reciprocal_rank_fusion"


"""
Configuration for the code ingestion pipeline.
Manages OpenSearch connection and embedding settings with Haystack-compatible schema.
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

# Embedding Model
EMBEDDING_MODEL = "text-embedding-3-small"

# Haystack-compatible field mappings
HAYSTACK_FIELD_MAPPING = {
    "embedding_field": "embedding",
    "text_field": "content",
    "metadata_field": "meta",
}

# Code Splitter Settings
CODE_SPLITTER_CONFIG = {
    "chunk_lines": 40,
    "chunk_lines_overlap": 15,
    "max_chars": 1500,
}

# Text Splitter Settings
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1024,
    "chunk_overlap": 200,
}

# File Processing Settings
MAX_FILE_SIZE_MB = 1  # Skip files larger than this


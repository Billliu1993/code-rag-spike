"""
Hybrid retrieval pipeline using Haystack.
Reads from OpenSearch index populated by LlamaIndex ingestion.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables BEFORE importing Haystack (for telemetry settings)
load_dotenv()
os.environ['HAYSTACK_TELEMETRY_ENABLED'] = 'False'

from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils import Secret
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever, OpenSearchBM25Retriever

# Add parent directory to import root config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OPENSEARCH_HOST,
    OPENSEARCH_USER,
    OPENSEARCH_PASSWORD,
    OPENSEARCH_INDEX,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    HAYSTACK_FIELD_MAPPING,
    DEFAULT_TOP_K_BM25,
    DEFAULT_TOP_K_EMBEDDING,
)


def get_document_store() -> OpenSearchDocumentStore:
    """
    Initialize OpenSearch document store with same configuration as ingestion.
    
    Returns:
        OpenSearchDocumentStore configured to read from LlamaIndex-populated index
    """
    # Parse host URL to extract hostname and port
    # OPENSEARCH_HOST format: "https://localhost:9200"
    host_parts = OPENSEARCH_HOST.replace("https://", "").replace("http://", "").split(":")
    host = host_parts[0]
    port = int(host_parts[1]) if len(host_parts) > 1 else 9200
    
    document_store = OpenSearchDocumentStore(
        hosts=[host],
        port=port,
        index=OPENSEARCH_INDEX,
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,  # For local development
        embedding_field=HAYSTACK_FIELD_MAPPING["embedding_field"],
        content_field=HAYSTACK_FIELD_MAPPING["content_field"],
        # Note: Using "metadata" to match what LlamaIndex actually stores
    )
    
    return document_store


def create_hybrid_pipeline(
    top_k_bm25: int = DEFAULT_TOP_K_BM25,
    top_k_embedding: int = DEFAULT_TOP_K_EMBEDDING,
) -> Pipeline:
    """
    Create hybrid retrieval pipeline combining BM25 and semantic search.
    
    Args:
        top_k_bm25: Number of results from BM25 (keyword) search
        top_k_embedding: Number of results from semantic (embedding) search
        
    Returns:
        Configured Haystack Pipeline
    """
    # Initialize document store
    document_store = get_document_store()
    
    # Create text embedder (same model as ingestion)
    text_embedder = OpenAITextEmbedder(
        model=EMBEDDING_MODEL,
        api_key=Secret.from_token(OPENAI_API_KEY) if OPENAI_API_KEY else Secret.from_env_var("OPENAI_API_KEY")
    )
    
    # Create BM25 retriever (keyword search)
    bm25_retriever = OpenSearchBM25Retriever(
        document_store=document_store,
        top_k=top_k_bm25,
    )
    
    # Create embedding retriever (semantic search)
    embedding_retriever = OpenSearchEmbeddingRetriever(
        document_store=document_store,
        top_k=top_k_embedding,
    )
    
    # Build pipeline
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("bm25_retriever", bm25_retriever)
    pipeline.add_component("embedding_retriever", embedding_retriever)
    
    # Connect components
    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    
    return pipeline


def query_pipeline(
    pipeline: Pipeline,
    query: str,
) -> dict:
    """
    Run a query through the hybrid retrieval pipeline.
    
    Args:
        pipeline: Configured Haystack pipeline
        query: Search query text
        
    Returns:
        Dictionary containing results from both retrievers
    """
    result = pipeline.run({
        "text_embedder": {"text": query},
        "bm25_retriever": {"query": query},
    })
    
    return result


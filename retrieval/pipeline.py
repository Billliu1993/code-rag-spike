"""
Hybrid retrieval pipeline using Haystack.
Reads from OpenSearch index populated by LlamaIndex ingestion.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# Load environment variables BEFORE importing Haystack (for telemetry settings)
load_dotenv()
os.environ['HAYSTACK_TELEMETRY_ENABLED'] = 'False'

from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils import Secret
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.components.retrievers.opensearch import OpenSearchHybridRetriever

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
    DEFAULT_JOIN_MODE,
    TRACEL_API_KEY,
)

# Initialize Traceloop for tracing (MUST be first to set up tracer provider)
# Traceloop.init(app_name="retrieval", disable_batch=True, telemetry_enabled=False, api_key=TRACEL_API_KEY)

# Integrate Python logging with OpenTelemetry for trace correlation
# set_logging_format=True calls logging.basicConfig() with trace context format
# IMPORTANT: Must be called BEFORE any logging.basicConfig() or getLogger() calls
LoggingInstrumentor().instrument(
    set_logging_format=True,
    log_level=logging.WARNING  # Set to WARNING for testing
)

# Get logger AFTER instrumentation
logger = logging.getLogger(__name__)

@task(name="connect_to_db")
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

@task(name="create_pipeline")
def create_hybrid_pipeline(
    top_k_bm25: int = DEFAULT_TOP_K_BM25,
    top_k_embedding: int = DEFAULT_TOP_K_EMBEDDING,
) -> Pipeline:
    """
    Create hybrid retrieval pipeline using OpenSearchHybridRetriever.
    
    Args:
        top_k_bm25: Number of results from BM25 (keyword) search
        top_k_embedding: Number of results from semantic (embedding) search
        
    Returns:
        Configured Haystack Pipeline with OpenSearchHybridRetriever
    """
    # Initialize document store
    document_store = get_document_store()
    
    # Create text embedder (same model as ingestion)
    embedder = OpenAITextEmbedder(
        model=EMBEDDING_MODEL,
        api_key=Secret.from_token(OPENAI_API_KEY) if OPENAI_API_KEY else Secret.from_env_var("OPENAI_API_KEY")
    )
    
    # Create hybrid retriever (combines BM25 and embedding search)
    hybrid_retriever = OpenSearchHybridRetriever(
        document_store=document_store,
        embedder=embedder,
        top_k_bm25=top_k_bm25,
        top_k_embedding=top_k_embedding,
        join_mode=DEFAULT_JOIN_MODE,
    )
    
    # Build simple pipeline
    pipeline = Pipeline()
    pipeline.add_component("retriever", hybrid_retriever)
    
    return pipeline

@task(name="run_query")
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
        Dictionary containing merged results from hybrid retriever
    """
    result = pipeline.run({
        "retriever": {"query": query},
    })
    logger.warning("Query pipeline completed")
    return result


"""
Hayhooks PipelineWrapper for hybrid code retrieval.
Deploys the retrieval pipeline as an MCP tool for IDE integration.
"""
import os
from typing import Any, Dict, List
from dotenv import load_dotenv

from hayhooks import BasePipelineWrapper
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils import Secret
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.components.retrievers.opensearch import OpenSearchHybridRetriever

# Load environment variables
load_dotenv()

# Configuration from environment (self-contained for deployment)
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "https://localhost:9200")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "code_index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
HAYSTACK_FIELD_MAPPING = {
    "embedding_field": "embedding",
    "content_field": "content",
}


class PipelineWrapper(BasePipelineWrapper):
    """
    Hybrid code retrieval using BM25 and semantic search.
    """
    
    def setup(self) -> None:
        """Initialize configuration for the hybrid retrieval pipeline."""
        # Parse host URL to extract hostname and port  
        host_parts = OPENSEARCH_HOST.replace("https://", "").replace("http://", "").split(":")
        self.host = host_parts[0]
        self.port = int(host_parts[1]) if len(host_parts) > 1 else 9200
    
    def _format_results(self, docs: List[Any]) -> List[Dict[str, Any]]:
        """
        Format results from hybrid retriever.
        
        Args:
            docs: Documents from hybrid retriever (already merged)
            
        Returns:
            Formatted list of results
        """
        formatted = []
        
        for i, doc in enumerate(docs, 1):
            content = doc.content if hasattr(doc, 'content') else ""
            score = doc.score if hasattr(doc, 'score') else 0.0
            meta_raw = doc.meta if hasattr(doc, 'meta') else {}
            
            # LlamaIndex nests metadata under a 'metadata' key
            metadata = meta_raw.get('metadata', meta_raw) if isinstance(meta_raw, dict) else {}
            
            formatted.append({
                "rank": i,
                "file_path": metadata.get("file_path", "Unknown"),
                "file_name": metadata.get("file_name", "Unknown"),
                "language": metadata.get("language", ""),
                "content_type": metadata.get("content_type", "text"),
                "content": content,  # Return full content, no truncation
                "score": round(score, 3),
            })
        
        return formatted
    
    def run_api(
        self, 
        query: str, 
        top_k_bm25: int = 3, 
        top_k_embedding: int = 3
    ) -> Dict[str, Any]:
        """
        Search code repository using hybrid retrieval (BM25 + semantic search).
        
        Combines keyword-based BM25 search with semantic embedding search
        to find the most relevant code snippets for your query. BM25 finds
        exact keyword matches while semantic search understands conceptual meaning.
        
        Args:
            query: Search query or question about the code
            top_k_bm25: Number of results from keyword search (default: 3)
            top_k_embedding: Number of results from semantic search (default: 3)
            
        Returns:
            Dictionary with merged results including file paths, full content, and scores
        """
        # Create fresh document store instance (components can't be shared between pipelines)
        document_store = OpenSearchDocumentStore(
            hosts=[self.host],
            port=self.port,
            index=OPENSEARCH_INDEX,
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
            use_ssl=True,
            verify_certs=False,
            embedding_field=HAYSTACK_FIELD_MAPPING["embedding_field"],
            content_field=HAYSTACK_FIELD_MAPPING["content_field"],
        )
        
        # Create fresh embedder instance
        embedder = OpenAITextEmbedder(
            model=EMBEDDING_MODEL,
            api_key=Secret.from_token(OPENAI_API_KEY) if OPENAI_API_KEY else Secret.from_env_var("OPENAI_API_KEY")
        )
        
        # Create hybrid retriever with dynamic top_k values
        hybrid_retriever = OpenSearchHybridRetriever(
            document_store=document_store,
            embedder=embedder,
            top_k_bm25=top_k_bm25,
            top_k_embedding=top_k_embedding,
            join_mode="reciprocal_rank_fusion",
        )
        
        # Build pipeline
        pipeline = Pipeline()
        pipeline.add_component("retriever", hybrid_retriever)
        
        # Run the pipeline
        result = pipeline.run({
            "retriever": {"query": query},
        })
        
        # Extract and format results
        docs = result.get("retriever", {}).get("documents", [])
        formatted_results = self._format_results(docs)
        
        # Return as JSON string for MCP compatibility
        import json
        response = {
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        return json.dumps(response, indent=2)


"""
CLI tool for querying the hybrid retrieval pipeline.
"""
import sys
from pathlib import Path
import click
from typing import List, Dict, Any

# Add parent directory to import root config
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.pipeline import create_hybrid_pipeline, query_pipeline
from config import DEFAULT_TOP_K_BM25, DEFAULT_TOP_K_EMBEDDING


def format_document(doc: Any, index: int, show_scores: bool = False, show_metadata: bool = False) -> str:
    """
    Format a single document for display.
    
    Args:
        doc: Document object from Haystack
        index: Document position in results
        show_scores: Whether to show relevance scores
        show_metadata: Whether to show full metadata
        
    Returns:
        Formatted string for display
    """
    content = doc.content if hasattr(doc, 'content') else ""
    meta_raw = doc.meta if hasattr(doc, 'meta') else {}
    score = doc.score if hasattr(doc, 'score') else 0.0
    
    # LlamaIndex nests metadata under a 'metadata' key
    metadata = meta_raw.get('metadata', meta_raw) if isinstance(meta_raw, dict) else {}
    
    # Extract key metadata
    file_path = metadata.get("file_path", "Unknown")
    file_name = metadata.get("file_name", "Unknown")
    content_type = metadata.get("content_type", "text")
    language = metadata.get("language", "")
    
    # Build header
    header_parts = [f"[{index}] {file_path}"]
    if language:
        header_parts.append(f"({language})")
    if show_scores:
        header_parts.append(f"| Score: {score:.3f}")
    
    header = " ".join(header_parts)
    
    # Truncate content if too long (unless showing full metadata)
    max_content_length = 500 if not show_metadata else 2000
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."
    
    # Build output
    lines = [
        "━" * 80,
        header,
        "━" * 80,
        content,
    ]
    
    if show_metadata:
        lines.append("")
        lines.append("Metadata:")
        for key, value in metadata.items():
            if key != "_node_content" and key != "_node_type":  # Skip LlamaIndex internal fields
                lines.append(f"  • {key}: {value}")
    
    return "\n".join(lines)


def merge_results(bm25_docs: List[Any], embedding_docs: List[Any]) -> List[Any]:
    """
    Merge and deduplicate results from BM25 and embedding retrievers.
    
    Args:
        bm25_docs: Documents from BM25 retriever
        embedding_docs: Documents from embedding retriever
        
    Returns:
        Merged and deduplicated list of documents
    """
    seen_ids = set()
    merged = []
    
    # Add BM25 results first (they might have exact keyword matches)
    for doc in bm25_docs:
        doc_id = doc.id if hasattr(doc, 'id') else str(getattr(doc.meta, 'document_id', ''))
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged.append(doc)
    
    # Add embedding results
    for doc in embedding_docs:
        doc_id = doc.id if hasattr(doc, 'id') else str(getattr(doc.meta, 'document_id', ''))
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged.append(doc)
    
    return merged


@click.command()
@click.option('--query', '-q', required=True, help='Search query text')
@click.option('--top-k-bm25', default=DEFAULT_TOP_K_BM25, help='Number of BM25 (keyword) results')
@click.option('--top-k-embedding', default=DEFAULT_TOP_K_EMBEDDING, help='Number of semantic (embedding) results')
@click.option('--show-scores', is_flag=True, help='Show relevance scores')
@click.option('--show-metadata', is_flag=True, help='Show full metadata for each result')
def main(query: str, top_k_bm25: int, top_k_embedding: int, show_scores: bool, show_metadata: bool):
    """
    Query the hybrid retrieval pipeline.
    
    Combines BM25 (keyword) and semantic (embedding) search for best results.
    """
    if not query.strip():
        click.echo("Error: Query cannot be empty", err=True)
        return
    
    click.echo(f"🔍 Query: \"{query}\"")
    click.echo()
    
    try:
        # Create pipeline
        click.echo("🔧 Initializing pipeline...")
        pipeline = create_hybrid_pipeline(
            top_k_bm25=top_k_bm25,
            top_k_embedding=top_k_embedding
        )
        
        # Run query
        click.echo("⚡ Running query...")
        result = query_pipeline(pipeline, query)
        
        # Extract results
        bm25_docs = result.get("bm25_retriever", {}).get("documents", [])
        embedding_docs = result.get("embedding_retriever", {}).get("documents", [])
        
        # Merge results
        merged_docs = merge_results(bm25_docs, embedding_docs)
        
        # Display summary
        click.echo()
        click.echo(f"📊 Found {len(merged_docs)} unique results ({len(bm25_docs)} BM25 + {len(embedding_docs)} Semantic)")
        click.echo()
        
        if not merged_docs:
            click.echo("❌ No results found")
            return
        
        # Display results
        for i, doc in enumerate(merged_docs, 1):
            formatted = format_document(doc, i, show_scores, show_metadata)
            click.echo(formatted)
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


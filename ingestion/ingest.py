"""
Main ingestion pipeline for code repositories into OpenSearch.
Uses LlamaIndex with Haystack-compatible schema.
"""
import os
import click
from pathlib import Path
from typing import List

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from llama_index.embeddings.openai import OpenAIEmbedding

from config import (
    OPENSEARCH_HOST,
    OPENSEARCH_USER,
    OPENSEARCH_PASSWORD,
    OPENSEARCH_INDEX,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    HAYSTACK_FIELD_MAPPING,
    CODE_SPLITTER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    MAX_FILE_SIZE_MB,
)
from utils import (
    detect_language,
    parse_gitignore,
    should_skip_file,
    check_file_should_process,
)


def get_vector_store() -> OpensearchVectorStore:
    """
    Initialize OpenSearch vector store with Haystack-compatible schema.
    
    Returns:
        OpensearchVectorStore instance configured for Haystack compatibility
    """
    from opensearchpy import OpenSearch
    
    # Create OpenSearch client for authentication
    os_client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False
    )
    
    # Create OpenSearch vector client with Haystack-compatible field names
    # text-embedding-3-small has dimension of 1536
    client = OpensearchVectorClient(
        endpoint=OPENSEARCH_HOST,
        index=OPENSEARCH_INDEX,
        dim=1536,  # Dimension for text-embedding-3-small
        embedding_field=HAYSTACK_FIELD_MAPPING["embedding_field"],
        text_field=HAYSTACK_FIELD_MAPPING["text_field"],
        os_client=os_client,
    )
    
    return OpensearchVectorStore(client)


def collect_files(repo_path: str, gitignore_patterns: set, verbose: bool = False) -> tuple[List[str], dict]:
    """
    Collect all files from repository that should be processed.
    
    Args:
        repo_path: Root path of the repository
        gitignore_patterns: Set of gitignore patterns to respect
        verbose: If True, log skipped files
        
    Returns:
        Tuple of (files_to_process, skip_stats)
        - files_to_process: List of file paths to process
        - skip_stats: Dictionary with skip reasons as keys and counts as values
    """
    files_to_process = []
    skip_stats = {}
    
    for root, dirs, files in os.walk(repo_path):
        # Filter directories in-place to skip ignored directories
        dirs[:] = [
            d for d in dirs
            if not should_skip_file(os.path.join(root, d), repo_path, gitignore_patterns)
        ]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check if file should be processed
            should_process, skip_reason = check_file_should_process(
                file_path, repo_path, gitignore_patterns, MAX_FILE_SIZE_MB
            )
            
            if should_process:
                files_to_process.append(file_path)
            else:
                # Track skip statistics
                skip_stats[skip_reason] = skip_stats.get(skip_reason, 0) + 1
                
                # Log if verbose
                if verbose:
                    rel_path = os.path.relpath(file_path, repo_path)
                    click.echo(f"⏭️  Skipped ({skip_reason}): {rel_path}")
    
    return files_to_process, skip_stats


def process_file(file_path: str, repo_path: str, repo_name: str, vector_store: OpensearchVectorStore) -> bool:
    """
    Process a single file and ingest into OpenSearch.
    
    Args:
        file_path: Absolute path to the file
        repo_path: Root path of the repository
        repo_name: Name of the repository
        vector_store: OpenSearch vector store instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip empty files
        if not content.strip():
            return True
        
        # Get relative path and filename
        rel_path = str(Path(file_path).relative_to(repo_path))
        file_name = os.path.basename(file_path)
        
        # Detect language and content type
        is_code, language = detect_language(file_path)
        
        # Create document with metadata
        metadata = {
            "file_path": rel_path,
            "file_name": file_name,
            "repo_name": repo_name,
            "content_type": "code" if is_code else "text",
        }
        
        if is_code and language:
            metadata["language"] = language
        
        doc = Document(text=content, metadata=metadata)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Select appropriate splitter
        if is_code and language:
            splitter = CodeSplitter(
                language=language,
                chunk_lines=CODE_SPLITTER_CONFIG["chunk_lines"],
                chunk_lines_overlap=CODE_SPLITTER_CONFIG["chunk_lines_overlap"],
                max_chars=CODE_SPLITTER_CONFIG["max_chars"],
            )
        else:
            splitter = SentenceSplitter(
                chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
                chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
            )
        
        # Create index and ingest (this will add embeddings automatically)
        VectorStoreIndex.from_documents(
            [doc],
            storage_context=storage_context,
            transformations=[splitter],
            show_progress=False,
        )
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


@click.command()
@click.option('--repo-path', required=True, type=click.Path(exists=True), help='Path to the repository to ingest')
@click.option('--verbose', is_flag=True, help='Show detailed logging including skipped files')
def main(repo_path: str, verbose: bool):
    """
    Ingest a code repository into OpenSearch for RAG.
    
    This pipeline:
    - Respects .gitignore patterns
    - Detects code languages automatically
    - Uses appropriate chunking strategies (CodeSplitter for code, SentenceSplitter for text)
    - Stores documents in Haystack-compatible schema
    """
    # Validate environment variables
    if not OPENAI_API_KEY:
        click.echo("Error: OPENAI_API_KEY environment variable not set", err=True)
        return
    
    # Configure embedding model globally
    Settings.embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )
    
    # Get repository name from path
    repo_name = os.path.basename(os.path.abspath(repo_path))
    
    click.echo(f"🚀 Starting ingestion for repository: {repo_name}")
    click.echo(f"📁 Repository path: {repo_path}")
    click.echo(f"🔍 Index: {OPENSEARCH_INDEX}")
    click.echo(f"🤖 Embedding model: {EMBEDDING_MODEL}")
    click.echo()
    
    # Parse gitignore
    click.echo("📋 Parsing .gitignore...")
    gitignore_patterns = parse_gitignore(repo_path)
    click.echo(f"   Found {len(gitignore_patterns)} patterns")
    
    # Collect files to process
    click.echo("📂 Collecting files...")
    if verbose:
        click.echo()
    files_to_process, skip_stats = collect_files(repo_path, gitignore_patterns, verbose)
    if verbose:
        click.echo()
    click.echo(f"   Found {len(files_to_process)} files to process")
    click.echo()
    
    if not files_to_process:
        click.echo("⚠️  No files to process!")
        return
    
    # Initialize vector store
    click.echo("🔌 Connecting to OpenSearch...")
    try:
        vector_store = get_vector_store()
        click.echo("   Connected successfully")
    except Exception as e:
        click.echo(f"❌ Failed to connect to OpenSearch: {e}", err=True)
        return
    
    click.echo()
    click.echo("⚙️  Processing files...")
    
    # Process files with progress tracking
    successful = 0
    failed = 0
    
    with click.progressbar(files_to_process, label='Ingesting files') as files:
        for file_path in files:
            if process_file(file_path, repo_path, repo_name, vector_store):
                successful += 1
            else:
                failed += 1
    
    # Summary
    click.echo()
    click.echo("=" * 60)
    click.echo("📊 Ingestion Summary")
    click.echo("=" * 60)
    click.echo(f"✅ Successfully processed: {successful} files")
    click.echo(f"❌ Failed: {failed} files")
    click.echo(f"📈 Total attempted: {len(files_to_process)} files")
    if len(files_to_process) > 0:
        click.echo(f"🎯 Success rate: {(successful/len(files_to_process)*100):.1f}%")
    
    # Show skip statistics
    if skip_stats:
        click.echo()
        click.echo("⏭️  Skipped files:")
        for reason, count in sorted(skip_stats.items()):
            click.echo(f"   • {reason}: {count} files")
        total_skipped = sum(skip_stats.values())
        click.echo(f"   • Total skipped: {total_skipped} files")
    
    click.echo("=" * 60)


if __name__ == "__main__":
    main()


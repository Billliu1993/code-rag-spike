"""
Utility functions for file detection, gitignore parsing, and filtering.
"""
import os
from pathlib import Path
from typing import Optional, Set, Tuple
import fnmatch


def detect_language(file_path: str) -> tuple[bool, Optional[str]]:
    """
    Detect if file is code and what language.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (is_code, language) where language is None if not code
    """
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.rs': 'rust',
        '.go': 'go',
        '.cpp': 'cpp',
        '.c': 'c',
        '.php': 'php',
        '.rb': 'ruby',
    }
    
    ext = Path(file_path).suffix.lower()
    if ext in language_map:
        return True, language_map[ext]
    return False, None


def parse_gitignore(repo_path: str) -> Set[str]:
    """
    Parse .gitignore file and return set of patterns.
    
    Args:
        repo_path: Root path of the repository
        
    Returns:
        Set of gitignore patterns
    """
    gitignore_path = Path(repo_path) / ".gitignore"
    patterns = {".git", ".git/**", "uv.lock"}  # Always skip .git directory and uv.lock
    
    if gitignore_path.exists():
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        patterns.add(line)
        except Exception as e:
            print(f"Warning: Could not read .gitignore: {e}")
    
    return patterns


def should_skip_file(file_path: str, repo_path: str, gitignore_patterns: Set[str]) -> bool:
    """
    Check if file should be skipped based on gitignore patterns.
    
    Args:
        file_path: Absolute path to the file
        repo_path: Root path of the repository
        gitignore_patterns: Set of gitignore patterns
        
    Returns:
        True if file should be skipped, False otherwise
    """
    # Get relative path from repo root
    try:
        rel_path = Path(file_path).relative_to(repo_path)
        rel_path_str = str(rel_path)
        rel_path_posix = rel_path.as_posix()  # Use forward slashes for pattern matching
    except ValueError:
        # File is outside repo path
        return True
    
    # Check against gitignore patterns
    for pattern in gitignore_patterns:
        # Handle directory patterns
        if pattern.endswith('/'):
            pattern = pattern.rstrip('/')
            if rel_path_posix.startswith(pattern + '/') or str(rel_path.parts[0]) == pattern:
                return True
        
        # Handle wildcard patterns
        if '*' in pattern or '?' in pattern:
            if fnmatch.fnmatch(rel_path_posix, pattern):
                return True
            if fnmatch.fnmatch(rel_path_str, pattern):
                return True
            # Check if any parent directory matches
            for part in rel_path.parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        else:
            # Exact match
            if rel_path_posix == pattern or rel_path_str == pattern:
                return True
            # Check if file is in ignored directory
            if pattern in rel_path.parts:
                return True
    
    return False


def check_file_should_process(file_path: str, repo_path: str, gitignore_patterns: Set[str], max_file_size_mb: float) -> Tuple[bool, Optional[str]]:
    """
    Check if file should be processed and return reason if it should be skipped.
    
    Args:
        file_path: Absolute path to the file
        repo_path: Root path of the repository
        gitignore_patterns: Set of gitignore patterns
        max_file_size_mb: Maximum file size in MB
        
    Returns:
        Tuple of (should_process, skip_reason)
        - should_process: True if file should be processed
        - skip_reason: Reason for skipping (None if should be processed)
    """
    # Check gitignore
    if should_skip_file(file_path, repo_path, gitignore_patterns):
        return False, "gitignore"
    
    # Check if binary
    if is_binary_file(file_path):
        return False, "binary"
    
    # Check file size
    file_size = get_file_size_mb(file_path)
    if file_size > max_file_size_mb:
        return False, f"too_large ({file_size:.1f}MB)"
    
    return True, None


def is_binary_file(file_path: str) -> bool:
    """
    Check if file is binary by attempting to read it as text.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file appears to be binary, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to read first 8KB
            f.read(8192)
        return False
    except (UnicodeDecodeError, PermissionError):
        return True


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    return os.path.getsize(file_path) / (1024 * 1024)


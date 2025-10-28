from llama_index.core.schema import Document, TextNode
from typing import List, Optional, Dict, Set, Tuple
from tree_sitter import Language, Parser, Node as TSNode
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_java
import tree_sitter_go
import tree_sitter_rust


class TreeSitterContextEnricher:
    """Post-processor that enriches LlamaIndex CodeSplitter nodes with context"""
    
    LANGUAGE_CONFIG = {
        'python': {
            'parser': tree_sitter_python,
            'import_types': ['import_statement', 'import_from_statement'],
            'class_types': ['class_definition'],
            'function_types': ['function_definition'],
            'identifier_types': ['identifier'],
        },
        'javascript': {
            'parser': tree_sitter_javascript,
            'import_types': ['import_statement'],
            'class_types': ['class_declaration'],
            'function_types': ['function_declaration', 'method_definition'],
            'identifier_types': ['identifier'],
        },
        'typescript': {
            'parser': tree_sitter_typescript.language_typescript(),
            'import_types': ['import_statement'],
            'class_types': ['class_declaration'],
            'function_types': ['function_declaration', 'method_definition'],
            'identifier_types': ['identifier'],
        },
        'java': {
            'parser': tree_sitter_java,
            'import_types': ['import_declaration'],
            'class_types': ['class_declaration'],
            'function_types': ['method_declaration'],
            'identifier_types': ['identifier'],
        },
        'go': {
            'parser': tree_sitter_go,
            'import_types': ['import_declaration'],
            'class_types': ['type_declaration'],
            'function_types': ['function_declaration', 'method_declaration'],
            'identifier_types': ['identifier'],
        },
        'rust': {
            'parser': tree_sitter_rust,
            'import_types': ['use_declaration'],
            'class_types': ['struct_item', 'enum_item'],
            'function_types': ['function_item'],
            'identifier_types': ['identifier'],
        },
    }
    
    def __init__(
        self,
        language: str = "python",
        include_imports: bool = True,
        include_parent_context: bool = True,
        smart_import_filtering: bool = True,
    ):
        """
        Initialize the enricher.
        
        Args:
            language: Programming language (python, javascript, typescript, java, go, rust)
            include_imports: Add relevant import statements to chunks
            include_parent_context: Add parent class/function signatures
            smart_import_filtering: Only include imports that are used in the chunk
        """
        self.language = language.lower()
        self.include_imports = include_imports
        self.include_parent_context = include_parent_context
        self.smart_import_filtering = smart_import_filtering
        
        if self.language not in self.LANGUAGE_CONFIG:
            raise ValueError(f"Unsupported language: {self.language}")
        
        self.config = self.LANGUAGE_CONFIG[self.language]
        self.parser = self._init_parser()
    
    def _init_parser(self) -> Parser:
        """Initialize tree-sitter parser"""
        parser_module = self.config['parser']
        
        # Handle different tree-sitter package structures
        if hasattr(parser_module, 'language'):
            language = parser_module.language()
        elif callable(parser_module):
            language = parser_module()
        else:
            language = Language(parser_module)
        
        parser = Parser()
        parser.set_language(language)
        return parser
    
    def enrich_nodes(
        self,
        nodes: List[TextNode],
        documents: List[Document]
    ) -> List[TextNode]:
        """
        Enrich LlamaIndex nodes with context.
        
        Args:
            nodes: Nodes from LlamaIndex CodeSplitter
            documents: Original source documents
        
        Returns:
            Enriched nodes with imports and parent context added
        """
        
        # Parse all documents once with tree-sitter
        parsed_docs = self._parse_documents(documents)
        
        # Enrich each node
        enriched_nodes = []
        for node in nodes:
            enriched_node = self._enrich_node(node, parsed_docs)
            enriched_nodes.append(enriched_node)
        
        return enriched_nodes
    
    def _parse_documents(self, documents: List[Document]) -> Dict:
        """Parse all documents with tree-sitter"""
        parsed = {}
        for doc in documents:
            tree = self.parser.parse(bytes(doc.text, "utf8"))
            parsed[doc.doc_id] = {
                'tree': tree,
                'text': doc.text,
                'bytes': bytes(doc.text, "utf8"),
                'lines': doc.text.split('\n')
            }
        return parsed
    
    def _enrich_node(self, node: TextNode, parsed_docs: Dict) -> TextNode:
        """Enrich a single node with context"""
        
        # Find source document
        doc_id = node.ref_doc_id or node.metadata.get('doc_id')
        if not doc_id or doc_id not in parsed_docs:
            return node
        
        doc_data = parsed_docs[doc_id]
        
        # Get line range for this node
        start_line = node.metadata.get('start_line', 0)
        end_line = node.metadata.get('end_line', start_line)
        
        if start_line == 0:
            start_line, end_line = self._find_node_lines(node.text, doc_data['lines'])
        
        # Collect context
        context_parts = []
        
        if self.include_imports:
            imports = self._extract_imports(
                doc_data['tree'].root_node,
                doc_data['bytes'],
                node.text if self.smart_import_filtering else None
            )
            if imports:
                context_parts.append(imports)
        
        if self.include_parent_context:
            parent_context = self._extract_parent_context(
                doc_data['tree'].root_node,
                doc_data['bytes'],
                start_line,
                end_line
            )
            if parent_context:
                context_parts.append(parent_context)
        
        # Return enriched node if we have context
        if not context_parts:
            return node
        
        enriched_text = "\n\n".join(context_parts + [node.text])
        
        return TextNode(
            text=enriched_text,
            id_=node.id_,
            embedding=node.embedding,
            metadata={
                **node.metadata,
                'original_code': node.text,
                'context_enriched': True,
            },
            excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
            relationships=node.relationships,
        )
    
    def _find_node_lines(self, node_text: str, source_lines: List[str]) -> Tuple[int, int]:
        """Find line range of node in source"""
        node_lines = [l.strip() for l in node_text.split('\n') if l.strip()]
        if not node_lines:
            return 0, 0
        
        first_line = node_lines[0]
        for i, line in enumerate(source_lines):
            if first_line in line.strip():
                return i, i + len(node_text.split('\n')) - 1
        
        return 0, 0
    
    def _extract_imports(
        self,
        root_node: TSNode,
        source_bytes: bytes,
        filter_text: Optional[str] = None
    ) -> Optional[str]:
        """Extract import statements"""
        import_types = self.config['import_types']
        imports = []
        
        def collect_imports(node: TSNode):
            if node.type in import_types:
                import_text = source_bytes[node.start_byte:node.end_byte].decode('utf8')
                
                if filter_text is None or self._is_import_relevant(import_text, filter_text):
                    imports.append(import_text)
            
            for child in node.children:
                collect_imports(child)
        
        collect_imports(root_node)
        return "\n".join(imports) if imports else None
    
    def _is_import_relevant(self, import_text: str, chunk_text: str) -> bool:
        """Check if import is used in the chunk"""
        imported_names = self._extract_imported_identifiers(import_text)
        
        for name in imported_names:
            if f" {name}" in chunk_text or f"{name}(" in chunk_text or f"{name}." in chunk_text:
                return True
        
        return False
    
    def _extract_imported_identifiers(self, import_text: str) -> Set[str]:
        """Extract identifier names from import"""
        tree = self.parser.parse(bytes(import_text, "utf8"))
        identifiers = set()
        
        def traverse(node: TSNode):
            if node.type in self.config['identifier_types']:
                name = import_text[node.start_byte:node.end_byte]
                # Avoid module names in dotted paths
                if node.parent and node.parent.type != 'dotted_name':
                    identifiers.add(name)
            
            for child in node.children:
                traverse(child)
        
        traverse(tree.root_node)
        return identifiers
    
    def _extract_parent_context(
        self,
        root_node: TSNode,
        source_bytes: bytes,
        start_line: int,
        end_line: int
    ) -> Optional[str]:
        """Extract parent class/function signatures"""
        
        # Find node containing this line range
        containing_node = self._find_containing_node(root_node, start_line, end_line)
        if not containing_node:
            return None
        
        # Walk up to find parent class/function
        current = containing_node.parent
        class_types = self.config['class_types']
        function_types = self.config['function_types']
        
        while current:
            if current.type in class_types or current.type in function_types:
                node_start = current.start_point[0]
                node_end = current.end_point[0]
                
                # Skip if this is the exact node we're enriching
                if node_start == start_line and node_end == end_line:
                    current = current.parent
                    continue
                
                # Extract signature
                signature = self._extract_signature(current, source_bytes)
                if signature:
                    indent = self._get_indent(signature)
                    return f"{signature}\n{indent}    # ..."
            
            current = current.parent
        
        return None
    
    def _find_containing_node(
        self,
        root_node: TSNode,
        start_line: int,
        end_line: int
    ) -> Optional[TSNode]:
        """Find smallest AST node containing the line range"""
        
        def search(node: TSNode) -> Optional[TSNode]:
            node_start = node.start_point[0]
            node_end = node.end_point[0]
            
            if node_start <= start_line and node_end >= end_line:
                # Try to find more specific child
                for child in node.children:
                    result = search(child)
                    if result:
                        return result
                return node
            
            return None
        
        return search(root_node)
    
    def _extract_signature(self, node: TSNode, source_bytes: bytes) -> Optional[str]:
        """Extract signature/declaration line"""
        full_text = source_bytes[node.start_byte:node.end_byte].decode('utf8')
        lines = full_text.split('\n')
        
        if self.language == 'python':
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('def ') or stripped.startswith('class '):
                    # Include decorators
                    start_idx = i
                    while start_idx > 0 and lines[start_idx - 1].strip().startswith('@'):
                        start_idx -= 1
                    
                    # Get signature up to colon
                    sig_lines = []
                    for j in range(start_idx, min(i + 3, len(lines))):
                        sig_lines.append(lines[j].rstrip())
                        if ':' in lines[j]:
                            break
                    
                    return '\n'.join(sig_lines)
        
        elif self.language in ['javascript', 'typescript', 'java']:
            sig_lines = []
            for line in lines[:3]:
                sig_lines.append(line.rstrip())
                if '{' in line:
                    break
            return '\n'.join(sig_lines)
        
        elif self.language == 'go':
            for line in lines[:3]:
                if 'func ' in line:
                    return line.rstrip()
        
        elif self.language == 'rust':
            for line in lines[:3]:
                if 'fn ' in line or 'struct ' in line:
                    return line.rstrip()
        
        return lines[0] if lines else None
    
    def _get_indent(self, text: str) -> str:
        """Get indentation from text"""
        if not text:
            return ""
        first_line = text.split('\n')[0]
        return first_line[:len(first_line) - len(first_line.lstrip())]



'''
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.schema import Document

# Your code
source_code = """
from utilities import format_complex
import math

class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
    
    def __str__(self):
        return format_complex(self.real, self.imag)
    
    def magnitude(self):
        return math.sqrt(self.real**2 + self.imag**2)
"""

# Step 1: Use LlamaIndex normally
documents = [Document(text=source_code)]
base_splitter = CodeSplitter.from_defaults(language="python")
base_nodes = base_splitter.get_nodes_from_documents(documents)

# Step 2: Enrich with context
enricher = TreeSitterContextEnricher(
    language="python",
    include_imports=True,
    include_parent_context=True,
    smart_import_filtering=True
)
enriched_nodes = enricher.enrich_nodes(base_nodes, documents)

# Check results
for i, node in enumerate(enriched_nodes):
    print(f"\n{'='*60}")
    print(f"Node {i+1}")
    print(f"{'='*60}")
    print(node.text)
'''

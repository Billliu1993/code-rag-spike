# Retrieval Pipeline Design

## Sequence Diagram

```mermaid
sequenceDiagram
    participant User as User (Actor)
    participant Copilot as GitHub Copilot
    participant MCP as MCP Tool
    participant EC2 as EC2 (Haystack + Hayhooks)
    participant OS as OpenSearch

    User->>Copilot: Work in IDE
    Copilot->>MCP: Trigger retrieval tool
    MCP->>EC2: Call Hayhooks endpoint
    EC2->>OS: Query (hybrid search)
    OS->>OS: Vector + keyword search
    OS-->>EC2: Return relevant code chunks
    EC2-->>MCP: Return results
    MCP-->>Copilot: Feed results back
    Copilot-->>User: Return results with code context
```

## Tech Stack

### Components

- **User**: Developer working in IDE
- **GitHub Copilot**: AI coding assistant with enhanced code context
- **MCP Tool**: Model Context Protocol tool for retrieval operations
- **EC2 (Haystack + Hayhooks)**: Hosted endpoint running Haystack retrieval pipeline with Hayhooks REST API
- **OpenSearch**: Vector database for hybrid search (semantic + lexical)

### Key Features

- Real-time code retrieval during development
- Hybrid search for accurate results
- Seamless integration with Copilot workflow
- Context-aware code suggestions
- Easy MCP integration via Hayhooks REST API endpoint on EC2


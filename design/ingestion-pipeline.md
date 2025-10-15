# Ingestion Pipeline Design

## Sequence Diagram

```mermaid
sequenceDiagram
    participant EB as EventBridge (Scheduled)
    participant ECS as ECS Fargate
    participant ECR as ECR (Container Registry)
    participant S3 as S3 Bucket
    participant Ingest as LlamaIndex Ingest Pipeline
    participant OS as OpenSearch

    EB->>ECS: Trigger Fargate task (scheduled event)
    ECS->>ECR: Pull container image
    ECR-->>ECS: Return image
    ECS->>ECS: Start task
    ECS->>S3: Fetch codebase
    S3-->>ECS: Return codebase data
    ECS->>Ingest: Process codebase
    Ingest->>Ingest: Parse & chunk code
    Ingest->>Ingest: Generate embeddings
    Ingest->>OS: Persist vectors & metadata
    OS-->>Ingest: Confirm storage
    Ingest-->>ECS: Complete
    ECS-->>EB: Task complete
```

## Tech Stack

### Components

- **EventBridge**: Scheduled trigger for ECS Fargate task execution
- **ECS Fargate**: Serverless container execution for long-running ingestion jobs (no time limits)
- **ECR**: Container registry for storing Docker images
- **S3**: Storage for raw codebase data
- **LlamaIndex Ingest Pipeline**: Integration framework for code processing and embedding
- **AWS Managed OpenSearch**: Managed search service with hybrid search capabilities (vector + keyword)

### Key Features

- Automated scheduled ingestion
- Scalable serverless architecture without time constraints
- Containerized deployment for flexibility
- Hybrid search support (semantic + lexical)
- Managed infrastructure for ease of setup


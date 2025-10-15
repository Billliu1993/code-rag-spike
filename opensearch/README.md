# OpenSearch Docker Setup

Simple Docker Compose setup for running OpenSearch locally, based on the [official Haystack configuration](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/opensearch/docker-compose.yml).

## Quick Start

```bash
cd opensearch
docker-compose up -d
```

Test the connection:
```bash
curl -k -u admin:admin https://localhost:9200
```

Default credentials: `admin` / `admin`

## Managing the Service

### Start
```bash
docker-compose up -d
```

### Stop (pause)
```bash
docker-compose stop
```

### Restart
```bash
docker-compose start
```

### Stop and remove
```bash
docker-compose down
```

### View logs
```bash
docker-compose logs -f opensearch
```

### Check status
```bash
docker-compose ps
```

## Configuration

### Ports
- `9200`: OpenSearch REST API
- `9600`: OpenSearch Performance Analyzer

### Memory Settings

Default: 1GB heap size (`-Xms1024m -Xmx1024m`)

To increase memory, modify the `ES_JAVA_OPTS` in docker-compose.yml:
```yaml
environment:
  - "ES_JAVA_OPTS=-Xms2048m -Xmx2048m"
```

## Python Integration

### Using with Haystack

```python
from haystack.document_stores import OpenSearchDocumentStore

document_store = OpenSearchDocumentStore(
    host="localhost",
    port=9200,
    username="admin",
    password="admin",
    scheme="https",
    verify_certs=False
)
```

### Using with opensearch-py

```python
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_auth=('admin', 'admin'),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)
```

## Index Management

### List all indices
```bash
curl -k -u admin:admin -X GET "https://localhost:9200/_cat/indices?v"
```

### Delete an index
```bash
# Delete a specific index (e.g., code_index)
curl -k -u admin:admin -X DELETE "https://localhost:9200/code_index?pretty"
```

### Verify index deletion
```bash
curl -k -u admin:admin -X GET "https://localhost:9200/_cat/indices?v"
```

### Count documents in an index
```bash
curl -k -u admin:admin -X GET "https://localhost:9200/code_index/_count?pretty"
```

### Search/query an index
```bash
# Get one sample document
curl -k -u admin:admin -X GET "https://localhost:9200/code_index/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "size": 1,
  "query": {
    "match_all": {}
  }
}
'
```

## Troubleshooting

### Container won't start
- Check if ports 9200/9600 are already in use: `lsof -i :9200`
- Increase Docker memory to at least 4GB

### Out of memory errors
- Adjust Java heap size in docker-compose.yml
- Increase Docker Desktop memory allocation

### Connection refused
- Wait for healthcheck to pass (up to 100 seconds)
- Check logs: `docker-compose logs opensearch`

## References

- [OpenSearch Documentation](https://opensearch.org/docs/latest/)
- [Haystack OpenSearch Integration](https://docs.haystack.deepset.ai/docs/opensearchdocumentstore)
- [Haystack docker-compose.yml](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/opensearch/docker-compose.yml)


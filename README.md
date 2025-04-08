# RAG model to query pdfs

Run a basic instance of Qdrant on docker
```
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

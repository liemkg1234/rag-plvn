from datetime import datetime
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from schema import RetrieverRequest
from services import RAGService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/vector_database/qdrant/get_collections", operation_id="get_collections")
async def get_collections() -> List[Dict]:
    """
    Get list collections/tables/indexes existed in vector database.

    Returns:
        collections (list): List of collections in the vector database.
            - id (str): ID of collection.
            - collection_name (str): Name of the collection.
            - description (str): Description of the collection.
    """
    return RAGService.get_collections()


@app.post("/rag/indexer", operation_id="rag_indexer")
async def indexer(
    collection_name: str = Form(..., description="Collection name to store documents in vector DB"),
    description: str = Form(..., description="Description of the collection"),
    files: List[UploadFile] = File(..., description="Upload one or more Markdown (.md) files only")):
    """
    Role: Indexer
    Team: RAG
    Description: Index document paths to vector database.

    Parameters:
        - collection_name (str): Collection name to save in vector database.
        - description (str): Description of the collection.
        - files (list): Paths to the source document.
    Return:
        - message (str): Success message.
        - collection_name (str): Collection name to save in vector database.
    """
    # Validate file types
    for file in files:
        if not file.filename.lower().endswith(".md"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a Markdown (.md) file.",
            )

    # Collection name with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond:06d}"

    collection = {
        "id": f"{collection_name}_{timestamp}",
        "collection_name": collection_name,
        "description": description,
    }

    # Indexer
    RAGService.index(collection, files)

    return {
        "message": "Index successful",
        "collection": collection,
    }


@app.post("/rag/retriever", operation_id="rag_retriever")
async def retrieve(request: RetrieverRequest):
    """
    Role: Retriever
    Team: RAG
    Description: Retriever document related with messages in vector database.

    Parameters: RetrieverRequest
        - question (str): User question.
        - collection_ids (list[str]): List collection id to save in vector database.
    Return:
        - documents_related (dict):
            - key: collection name
            - value: documents related to the collection.
    """
    # Check if requested collection exists
    collections = await get_collections()
    collection_ids = [c["id"] for c in collections]

    invalid_ids = [id_ for id_ in request.collection_ids if id_ not in collection_ids]

    if invalid_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Collection(s) not found: {invalid_ids}",
        )

    return RAGService.retrieve(request.question, request.collection_ids)


@app.post("/rag/chat", operation_id="rag_chat")
async def chat(request: RetrieverRequest):
    """
    Role: Chat
    Team: RAG
    Description: Chat with document related with messages in vector database.

    Parameters: RetrieverRequest
        - question (str): User question.
        - collection_ids (list[str]): List collection id to save in vector database.
    Return:
        - documents_related (dict):
            - key: collection name
            - value: documents related to the collection.
        - answer: The answer to the question.
    """

    return RAGService.chat(request.question, request.collection_ids)

mcp = FastApiMCP(
    app,
    exclude_operations=["rag_indexer"],
)
mcp.mount()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

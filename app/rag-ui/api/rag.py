import os

import httpx
from streamlit.runtime.uploaded_file_manager import UploadedFile

URL = os.getenv("RAG_BE_URL")


async def get_qdrant_collections():
    url = f"{URL}/vector_database/qdrant/get_collections"
    async with httpx.AsyncClient() as client:
        response = await client.post(url)
        response.raise_for_status()
        return response.json()


async def index_documents(collection_name: str, description: str, uploaded_files: list[UploadedFile]):
    url = f"{URL}/rag/indexer"
    files = []

    for file in uploaded_files:
        filename = file.name
        content = file.getvalue()

        files.append((
            "files",
            (filename, content, "text/markdown")
        ))

    data = {"collection_name": collection_name, "description": description}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=url,
            data=data,
            files=files,
            timeout=60*60.0,
        )

    return response


async def retriever(collection_ids: list, messages: list[dict], mode: bool = False):
    if not mode:
        # Retriever
        url = f"{URL}/rag/retriever"
    else:
        # Chat mode
        url = f"{URL}/rag/chat"

    # Get user input
    question = messages[-1]['content']

    data = {
        "question": question,
        "collection_ids": collection_ids,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=url,
            json=data,
            timeout=60*60.0,
        )
    return response.json()

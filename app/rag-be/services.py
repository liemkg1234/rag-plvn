import os
import shutil
from typing import Dict, List

from common.chunk import chunker
from common.qdrant import EmbedModel, QdrantService, RerankModel
from common.retrieve import retriever
from fastapi import UploadFile


class RAGService:
    __instance = None
    @staticmethod
    def get_collections() -> List[Dict]:
        """
        Get list collections/tables/indexes existed in vector database.
        """
        qdrant_service = QdrantService(url=os.getenv("QDRANT_DB_URL"))
        return qdrant_service.get_collections()


    @staticmethod
    def index(collection: dict, files: List[UploadFile]) -> None:
        """

        Parameters:
            :param collection:
            - collection (dict):
                id (str)
                collection_name: (str)
                description: (str)

            :param files:
            - files (list)

        Index document paths to vector database.
            1. Save files to folder local
            2. Index:
                - Chunk
                - Embed
                - Store to vector database

        """
        # Step 1: Create temp folder to store files
        folder_path = f"/tmp/{collection['id']}"
        os.makedirs(folder_path, exist_ok=True)

        try:
            for file in files:
                file_path = os.path.join(folder_path, file.filename)
                with open(file_path, "wb") as f:
                    content = file.file.read()
                    f.write(content)

            # Step 2: Chunk the files
            data_chunked = chunker(
                folder_path=folder_path,
                min_chunk_size=256,
                max_chunk_size=1024,
                context_retrieval=False,
                debug=True,
            )

            # Step 3: Index into vector DB
            embed_model = EmbedModel(
                api_base=os.getenv("LLM_GATEWAY_URL"),
                api_key=os.getenv("LLM_LAB_API_KEY"),
                model_name=os.getenv("EM_MODEL"),
                provider="openai",
            )

            qdrant_service = QdrantService(url=os.getenv("QDRANT_DB_URL"))

            qdrant_service.embed_index(
                embed_model=embed_model,
                data_chunked=data_chunked,
                collection=collection,
            )

        finally:
            # Step 4: Cleanup folder
            shutil.rmtree(folder_path, ignore_errors=True)

        return


    @staticmethod
    def retrieve(question: str, collections_name: List[str]) -> Dict:
        """
        Retrieve documents related to the question.

        Steps:
            1. Get store indexes from collections.
            2. Retrieve the nodes from the store index.
        """
        # Get store indexes from collections
        embed_model = EmbedModel(
            api_base=os.getenv("LLM_GATEWAY_URL"),
            api_key=os.getenv("LLM_LAB_API_KEY"),
            model_name=os.getenv("EM_MODEL"),
            provider="openai",
        )

        rerank_model = RerankModel(
            api_base=os.getenv("LLM_GATEWAY_URL"),
            api_key=os.getenv("LLM_LAB_API_KEY"),
            model_name=os.getenv("RM_MODEL"),
        )

        qdrant_service = QdrantService(url=os.getenv("QDRANT_DB_URL"))

        store_indexes = {}

        for collection_name in collections_name:
            # Get store index
            store_index = qdrant_service.get_store_index(embed_model=embed_model, collection_name=collection_name)

            # Retrieve
            document_related = retriever(
                store_index=store_index,
                question=question,

                # Cosine Similarity
                similarity_top_k=5,
                similarity_cutoff=0.1,

                # Rerank
                enable_rerank=False,
                rerank_client=rerank_model,
                top_n=5,

                debug=False,
            )

            store_indexes[collection_name] = document_related

        return {
            "document_related": store_indexes
        }

    @staticmethod
    def chat(question: str, collections_name: List[str]) -> Dict:
        # Get list document related
        document_related = RAGService.retrieve(question=question, collections_name=collections_name)['document_related']

        # Chat
        from textwrap import dedent

        from openai import OpenAI

        client = OpenAI(
            base_url=os.getenv("LLM_GATEWAY_URL"),
            api_key=os.getenv("LLM_LAB_API_KEY"),
        )

        completion = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {
                    "role": "system",
                    "content": dedent(f"""\
                        You are a Retrieval-Augmented Generation (RAG) chatbot.
                        Your role is to find the right answer from the document for the user question.

                        - Rules:
                        <rules>
                        - Always response with Markdown format.
                        - When you don't know the answer, say "I don't know".
                        </rules>
s
                        - Documents:
                        {document_related}
                    """)
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )

        answer = completion.choices[0].message.content

        print(answer)

        return {
            "document_related": document_related,
            "answer": answer
        }

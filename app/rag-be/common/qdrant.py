import datetime
import uuid
from typing import Dict, List

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.tei_rerank import TextEmbeddingInference
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


class EmbedModel:
    def __init__(self, api_base: str, model_name: str, api_key: str, provider: str = "openai"):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider

    def get_embed_model(self):
        if self.provider == "openai":
            return OpenAIEmbedding(api_base=self.api_base, model_name=self.model_name, api_key=self.api_key)
        if self.provider == "cohere":
            return CohereEmbedding(model_name=self.model_name,api_key=self.api_key)


class RerankModel:
    def __init__(self, api_base: str, model_name: str, api_key: str, provider: str = "openai"):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider

    def get_rerank_model(self, top_n: int = 5):
        if self.provider == "openai":
            return TextEmbeddingInference(
                base_url=self.api_base,
                auth_token=self.api_key,
                model_name=self.model_name,

                # Settings
                top_n=top_n,
            )

        elif self.provider == "cohere":
            return CohereRerank(
                model=self.model_name,
                api_key=self.api_key,

                # Settings
                top_n=top_n,
            )
        return None


class QdrantService:
    def __init__(self, url: str):
        self.client = QdrantClient(url=url)

    ###
    # Management collections with collection_metadata
    ###

    def _init_metadata_collection(self):
        """
        Using collection_metadata to store metadata of all collections.
        """
        collection_name = "collection_metadata"
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if collection_name not in existing:
            print(f"[+] Creating metadata collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )
        else:
            print(f"[i] Metadata collection '{collection_name}' already exists.")

    def _add_to_metadata(self, collection: dict):
        """
        Add new collection to collection_metadata.

        Parameters:
            - collection (dict):
                - id
                - collection_name
                - description
        """
        self._init_metadata_collection()

        self.client.upsert(
            collection_name="collection_metadata",
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[0.0],
                    payload={
                        "id": collection['id'],
                        "collection_name": collection['collection_name'],
                        "description": collection['description'],
                        "created_at": datetime.datetime.now().isoformat()
                    },
                )
            ]
        )
        print(f"[+] Added metadata for collection: {collection}")

    def get_collections(self) -> List[Dict]:
        """
        Get a list of all collections from collection_metadata with their descriptions.
        """
        self._init_metadata_collection()

        results = []
        response = self.client.scroll(
            collection_name="collection_metadata",
            limit=1000,
            with_payload=True,
        )

        for point in response[0]:
            payload = point.payload
            results.append(
                {
                    "id": payload.get("id"),
                    "collection_name": payload.get("collection_name"),
                    "description": payload.get("description")
                }
            )

        return results

    ###
    # Indexing & Retrieving
    ###

    def embed_index(
            self,
            embed_model: EmbedModel,
            data_chunked: list,
            collection: dict,
    ):
        vector_store = QdrantVectorStore(
            collection['id'],
            client=self.client,
            batch_size=20,
            enable_hybrid=False,
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )

        VectorStoreIndex(
            data_chunked,
            storage_context=storage_context,
            embed_model=embed_model.get_embed_model(),

            show_progress=True,
        )
        self._add_to_metadata(collection)

        print(f"Indexing complete with collection name: {collection}")

    def get_store_index(self, embed_model: EmbedModel, collection_name: str) -> VectorStoreIndex:
        vector_store = QdrantVectorStore(
            collection_name,
            client=self.client,
            batch_size=20,
            enable_hybrid=False,
        )

        store_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model.get_embed_model(),
        )
        return store_index

    def close(self):
        self.client.close()

from typing import List

from pydantic import BaseModel, Field


class RetrieverRequest(BaseModel):
    question: str = Field(..., description="Question to retrieve related documents")
    collection_ids: List[str] = Field(..., description="Collections id to store documents in vector DB")

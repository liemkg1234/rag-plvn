import copy
import re

from common.qdrant import RerankModel
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor


def retriever(
    store_index: VectorStoreIndex,
    question: str,
    similarity_top_k: int = 20,
    enable_similarity_cutoff: bool = True,
    similarity_cutoff: float = 0.5,
    enable_rerank: bool = True,
    rerank_client: RerankModel = None,
    top_n: int = 5,
    debug: bool = False,
) -> str:
    """
    Retrieve the documents related to the question.

    Steps:
        1. Retrieve the nodes from the store index.
        2. Post-process the nodes (Similarity cutoff, Rerank).
        3. Get the full content of the paragraphs in the metadata of the chunks.
    """
    # Retrieve
    retrieve = store_index.as_retriever(
        similarity_top_k=similarity_top_k,
    )
    nodes = retrieve.retrieve(question)
    if debug:
        print(f"Retrieved {len(nodes)} nodes")
        for node in nodes:
            print(f"Node: {node}")

    # Post-processors
    if enable_similarity_cutoff:
        # Similarity cutoff
        processor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        nodes = processor.postprocess_nodes(nodes)
    if enable_rerank:
        # Shorten text
        short_nodes = copy.deepcopy(nodes)
        for node in short_nodes:
            text = re.sub(r"-+", "-", node.text)
            text = re.sub(r"\s+", " ", text)
            node.node.text = text[:30000]
            node.node.metadata = {}

        # Rerank
        processor = rerank_client.get_rerank_model(top_n=top_n)
        reranked_short_nodes = processor.postprocess_nodes(short_nodes, query_str=question)

        # Filter nodes by node_id
        reranked_ids = [node.node_id for node in reranked_short_nodes]
        nodes = [node for node in nodes if node.node_id in reranked_ids]

    # Get full content of paragraphs in metadata of chunks
    chunks = []
    seen_ids = set()

    for node in nodes:
        paragraph_id = node.metadata.get("paragraph_id")
        full_content = f"Position: {node.metadata.get('file_path')}{node.metadata.get('header_path')}\n\nContent:\n {node.metadata.get('paragraph_full_content')}"

        if paragraph_id and full_content and paragraph_id not in seen_ids:
            seen_ids.add(paragraph_id)
            chunks.append(full_content)

    if not chunks:
        return """Information Not Found"""

    documents = "List Paragraph Related:\n"
    for i, chunk in enumerate(chunks):
        documents += f"""
<paragraph_{i+1}>

{chunk}

</paragraph_{i+1}>

"""

    return documents

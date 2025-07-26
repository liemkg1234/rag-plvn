import concurrent.futures
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import tiktoken
from common.context_retrieval import invoke
from common.sentence_splitter import CustomSentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import BaseNode, MetadataMode


def chunker(folder_path,
            min_chunk_size: int = 256,
            max_chunk_size: int = 1024,
            context_retrieval: bool = False,
            debug: bool = False
            ):
    """
    Chunk the documents in the folder_path into smaller chunks of size chunk_size.

    Steps:
        1. MarkdownNodeParser: Parse the markdown files into nodes.
        2. Merge small chunks: Merge small chunks that are less than min_size tokens.
        3. CustomSentenceSplitter: Split the chunks into sentences, and add full content of paragraph into metadata.
        4. Context Retrieval
    """
    files = sorted(glob.glob(os.path.join(folder_path, "**/*.md"), recursive=True))
    documents = SimpleDirectoryReader(input_files=files).load_data()

    # Markdown Splitter
    splitter = MarkdownNodeParser()

    pipeline = IngestionPipeline(transformations=[splitter])
    chunks = pipeline.run(documents=documents)

    # Merge small chunks
    chunks = merge_small_chunks(chunks, min_size=min_chunk_size)

    # Sentence Splitter
    splitter2 = CustomSentenceSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        separator="\n",
        paragraph_separator="\n\n\n",
        secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
    )
    chunks = splitter2(chunks)

    # Context Retrieval
    if context_retrieval:
        chunks = process_nodes(chunks, max_workers=5)

    # Debugging
    if debug:
        print("Length of chunks:", len(chunks))
        for chunk in chunks:
            print("=" * 80)
            print(f"📏 Tokens: {count_tokens(chunk.get_content())}")
            print(f"📄 Chunk:\n{chunk.get_content()}")

            print("\n🗂️ Metadata:")
            for key, value in chunk.metadata.items():
                print(f"  • {key}: {value}")

            print(f"\n🔗 Relationships: {chunk.relationships}")
            print("=" * 80)

    return chunks


def merge_nodes(node1, node2):
    return type(node1)(
        text=node1.get_content(metadata_mode=MetadataMode.NONE) + "\n\n" + node2.get_content(metadata_mode=MetadataMode.NONE),
        metadata=node1.metadata,
        relationships=node1.relationships,
    )


def merge_small_chunks(nodes: Sequence[BaseNode], min_size=100):
    nodes = list(nodes)
    i = 0

    while i < len(nodes):
        curr_node = nodes[i]
        curr_size = count_tokens(curr_node.get_content(metadata_mode=MetadataMode.NONE))

        if curr_size < min_size:
            prev_node = nodes[i - 1] if i > 0 else None
            next_node = nodes[i + 1] if i < len(nodes) - 1 else None
            prev_mergeable = (
                prev_node
                and prev_node.metadata["file_name"] == curr_node.metadata["file_name"]
            )
            next_mergeable = (
                next_node
                and next_node.metadata["file_name"] == curr_node.metadata["file_name"]
            )

            if prev_mergeable and next_mergeable:
                prev_size = count_tokens(prev_node.get_content(metadata_mode=MetadataMode.NONE))
                next_size = count_tokens(next_node.get_content(metadata_mode=MetadataMode.NONE))
                if prev_size <= next_size:
                    nodes[i - 1] = merge_nodes(prev_node, curr_node)
                    nodes.pop(i)
                    continue
                else:
                    nodes[i] = merge_nodes(curr_node, next_node)
                    nodes.pop(i + 1)
                    continue
            elif prev_mergeable:
                nodes[i - 1] = merge_nodes(prev_node, curr_node)
                nodes.pop(i)
                continue
            elif next_mergeable:
                nodes[i] = merge_nodes(curr_node, next_node)
                nodes.pop(i + 1)
                continue

        i += 1

    return nodes


def process_node(node):
    node.text = invoke(
        file_name=node.metadata['file_path'],
        whole_document=node.metadata,
        chunk_content=node.get_content().strip(),
        language="Vietnamese"
    )
    return node


def process_nodes(nodes, max_workers: int = 4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_node = {executor.submit(process_node, node): node for node in nodes}

        processed_nodes = []
        for future in concurrent.futures.as_completed(future_to_node):
            try:
                processed_node = future.result()
                processed_nodes.append(processed_node)
            except Exception as e:
                print(f"Generated an exception: {e}")

    return processed_nodes


# Common
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

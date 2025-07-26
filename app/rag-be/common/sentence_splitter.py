import hashlib
from typing import Any, List, Sequence

from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.utils import get_tqdm_iterable


class CustomSentenceSplitter(SentenceSplitter):
    """Custom SentenceSplitter that adds original text and source_id to metadata of split nodes."""

    def _generate_source_id(self, text: str) -> str:
        """Generate a unique id for the text using SHA-256."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _parse_nodes(
            self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            original_content = node.get_content(metadata_mode=MetadataMode.NONE)
            metadata_str = self._get_metadata_str(node)

            source_id = self._generate_source_id(original_content)

            splits = self.split_text_metadata_aware(
                original_content,
                metadata_str=metadata_str,
            )

            new_nodes = build_nodes_from_splits(splits, node, id_func=self.id_func)

            for new_node in new_nodes:
                new_node.metadata["paragraph_id"] = source_id
                new_node.metadata["paragraph_full_content"] = original_content

                # Don't add paragraph_full_content to embedding
                new_node.excluded_embed_metadata_keys.extend(['paragraph_id', 'paragraph_full_content'])

            all_nodes.extend(new_nodes)

        return all_nodes

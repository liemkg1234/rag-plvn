import streamlit as st
import streamlit_mermaid as stmd

st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.title("RAG Pipeline")


st.markdown("""
## Indexing

1. **Parse Document**: Chuyển dữ liệu về dạng Markdown (vì đơn giản, nhẹ, dễ đọc, linh hoạt) rất phù hợp cho LLM hiểu và xử lý
    - **Document Parsing**: Docling is the best
2. **Chunking**: Chia tài liệu thành các đoạn nhỏ hơn để dễ dàng xử lý, các bước:
    1. **Semantic Analysis**: Semantic chunking: Chia thành các đề mục
    2. **Consolidation**: Merge chunking: Gộp các đoạn nhỏ lại với nhau
    3. **Fine-grained Processing**: Sentence Splitter: Lưu chunk vào metadata, sau đó chia nó thành các đoạn lớn (ở step 1) thành các câu nhỏ hơn
    4. **Metadata Enrichment**: Context Retrieval: Tạo các thông tin (summary, header, keywords) cho từng chunk để cải thiện khả năng tìm kiếm
- **Optimal Size**: 256 < chunk_size < 2048 
3. **Embedding**: Dùng Embedding Models (EMs) để chuyển chunks thành các vector
    - **Multi-language Support**: Chọn các EMs có khả năng multi-language: Cohere (API), bge (hosting)
4. **Storage**: Lưu trữ các vector vào Vector Database (VD) để dễ dàng tìm kiếm, Qdrant vì:
    - **Advanced Search**: Hỗ trợ Hybrid Search (TF-IDF + Dense Vector) để cải thiện khả năng tìm kiếm
    - **Flexibility**: Hổ trợ local storage, có thể export ra SQL file để move

## Retrieval
1. **Pre-Retrieval**: Tiền xử lý câu hỏi (Để LLM Agent lo)
2. **Retrieval**: Tìm kiếm các vector gần nhất với câu hỏi
3. **Post-Retrieval**: Lọc các chunks
    - **Hybrid Search**: TF-IDF (BM25) + Dense Vector (Cosine Similarity)
    - **Filtering**: Cutoff & Top-n: Lọc các chunk không liên quan
    - **Reranking**: Rerank: Cohere hoặc bge-m3 với top-k
""")


mermaid_diagram = """
graph TD
    subgraph Indexing
        A[Parse Document] --> B[Chunking]
        B --> C[Embedding]
        C --> D[Storage]
    end

    subgraph Retrieval
        E[Pre-Retrieval] --> F[Retrieval]
        F --> G[Post-Retrieval]
        G --> H[Best Top-k Chunks]
    end

    D --> F
"""

st.markdown("# Diagram")
stmd.st_mermaid(mermaid_diagram)

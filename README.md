# RAG PLVN

## Bước 1. Tiền xử lý dữ liệu
- Chuyển dữ liệu dạng Word (.DOCX) về dạng Markdown (.MD)
- Lý do: vì dữ liệu dạng Markdown đơn giản, nhẹ, dễ đọc, linh hoạt rất phù hợp cho NLP, LLM hiểu và xử lý
- Thư viện sử dụng: Docling
- Dữ liệu được lưu ở: dataset/preprocessed/PLVN

## Bước 2. Xây dựng Hệ thống tim kiếm thông tin
### 1. Embedding Document
- Chunking: Dùng Llama-index (app/rag-be/common/chunk.py)
- Embedding & Indexing: Dùng model bge-m3 để embedding text, và dùng Qdrant database để lưu trữ
- Retrieval: Dùng Cosine để đo độ tương đồng giữa question - text embedded (top_p = 5, cosine_score = 0.5)

- Vị trí test:
    - Kết quả: test/results.csv
    - File test: test/test.py

## Bước 3. Triển khai API 

- API

```bash
curl -X 'POST' \
  'http://localhost:8001/rag/retriever' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "Người lao động bị sa thải có được trả lương hay không?",
  "collection_ids": [
    "Luat_45_2019_QH14_20250726091858830992"
  ]
}'
```
- Demo Test: images/RAG_PLVN.mp4
- Link demo: http://localhost:3000/Retrieval

# Cách sử dụng

### Bước 1. Install `just`
```bash
curl -fsSL https://just.systems/install.sh | bash -s -- --to /usr/local/bin
```

### Bước 2. Start server
```bash
just start
```


# Thông tin thêm
## Port Public
- Embedding Server: http://localhost:8000
- Document Parsing Server: http://localhost:9999/ui/
- Qdrant Server: http://localhost:6333/dashboard#/collections
- Rag-Backend: http://localhost:8001/docs
- Rag-UI: http://localhost:3000

## Model sử dụng:
https://huggingface.co/gpustack/bge-m3-GGUF

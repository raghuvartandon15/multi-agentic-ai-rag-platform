from pathlib import Path

MODEL_PROVIDER = "groq"
MODEL = "openai/gpt-oss-120b"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "my_chroma_db_v1"

BM25_DOCUMENTS_PATH = (
    PROJECT_ROOT / "rag" / "bm25_documents.json"
)


EMBEDDING_MODEL = (
    "sentence-transformers/all-mpnet-base-v2"
)

CROSS_ENCODER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

# Retrieval
DENSE_K = 8
BM25_K = 8

FINAL_K = 5

DENSE_WEIGHT = 0.6

BM25_WEIGHT = 0.4
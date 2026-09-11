from pathlib import Path
import json

from bs4 import SoupStrainer

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rag.config import (
    CHROMA_PATH,
    BM25_DOCUMENTS_PATH,
    EMBEDDING_MODEL,
)


# ==========================================
# Documentation URLs
# ==========================================

triton_urls = [
    "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html",
    "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html",
    "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/README.html",
]

mlflow_urls = [
    "https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/",
    "https://mlflow.org/docs/latest/self-hosting/troubleshooting/",
    "https://mlflow.org/docs/latest/ml/deployment/",
    "https://mlflow.org/docs/latest/ml/deployment/deploy-model-locally",
    "https://mlflow.org/docs/latest/ml/model-registry/tutorial",
]

kubernetes_urls = [
    "https://kubernetes.io/docs/tasks/debug/debug-application/",
    "https://kubernetes.io/docs/tasks/debug/debug-application/debug-running-pod/",
    "https://kubernetes.io/docs/tasks/debug/debug-application/debug-init-containers/",
]

aws_lambda_urls = [
    "https://docs.aws.amazon.com/lambda/latest/dg/lambda-troubleshooting.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/troubleshooting-invocation.html",
]

aws_ecs_urls = [
    "https://docs.aws.amazon.com/AmazonECS/latest/developerguide/troubleshooting.html",
    "https://docs.aws.amazon.com/AmazonECS/latest/developerguide/cannot-start-container.html",
    "https://docs.aws.amazon.com/AmazonECS/latest/developerguide/container-runtime-error.html",
]

fastapi_urls = [
    "https://fastapi.tiangolo.com/deployment/concepts/",
    "https://fastapi.tiangolo.com/deployment/server-workers/",
    "https://fastapi.tiangolo.com/deployment/docker/",
]


sources = {
    "triton": triton_urls,
    "mlflow": mlflow_urls,
    "kubernetes": kubernetes_urls,
    "aws_lambda": aws_lambda_urls,
    "aws_ecs": aws_ecs_urls,
    "fastapi": fastapi_urls,
}


# ==========================================
# Load Documents
# ==========================================

def load_documents():

    all_docs = []

    for technology, urls in sources.items():

        for url in urls:

            loader = WebBaseLoader(
                [url],
                bs_kwargs={
                    "parse_only": SoupStrainer("main")
                }
            )

            docs = loader.load()

            for doc in docs:

                doc.metadata["technology"] = technology

                doc.metadata["source_url"] = url

                doc.metadata["document_type"] = (
                    "official_documentation"
                )

                doc.metadata["title"] = doc.metadata.get(
                    "title",
                    f"{technology} documentation"
                )

            all_docs.extend(docs)

    print(
        f"Total documents loaded: {len(all_docs)}"
    )

    return all_docs


# ==========================================
# Create Chunks
# ==========================================

def create_chunks(
    documents,
    chunk_size=1000,
    chunk_overlap=150
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Total chunks created: {len(chunks)}"
    )

    return chunks


# ==========================================
# Save BM25 Documents
# ==========================================

def save_documents_for_bm25(chunks):

    documents = []

    for chunk in chunks:

        documents.append(
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
        )

    BM25_DOCUMENTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        BM25_DOCUMENTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Saved {len(documents)} chunks for BM25."
    )


# ==========================================
# Load BM25 Documents
# ==========================================

def load_documents_for_bm25():

    if not BM25_DOCUMENTS_PATH.exists():

        raise FileNotFoundError(
            f"\nBM25 document file not found:\n"
            f"{BM25_DOCUMENTS_PATH}\n\n"
            f"Run ingestion first."
        )

    with open(
        BM25_DOCUMENTS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        documents = json.load(f)

    return [
        Document(
            page_content=doc["page_content"],
            metadata=doc["metadata"]
        )
        for doc in documents
    ]


# ==========================================
# Build Vector Store
# ==========================================

def build_vectorstore():

    documents = load_documents()

    chunks = create_chunks(
        documents
    )

    # Save chunks for BM25
    save_documents_for_bm25(
        chunks
    )

    # Create embeddings
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Create Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=str(CHROMA_PATH)
    )

    print("Chroma vector store created.")

    return vectorstore


# ==========================================
# Run Ingestion
# ==========================================

if __name__ == "__main__":

    build_vectorstore()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from rag.config import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
    DENSE_K,
    BM25_K,
    FINAL_K,
    DENSE_WEIGHT,
    BM25_WEIGHT,
)

from rag.ingestion import (
    load_documents_for_bm25
)


class HybridRetriever:

    def __init__(self):

        print("Initializing RAG retriever...")

        # ==========================================
        # Embeddings
        # ==========================================

        self.embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        # ==========================================
        # Chroma
        # ==========================================

        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=self.embedder
        )

        self.dense_retriever = (
            self.vectorstore.as_retriever(
                search_kwargs={
                    "k": DENSE_K
                }
            )
        )

        # ==========================================
        # BM25
        # ==========================================

        bm25_documents = (
            load_documents_for_bm25()
        )

        self.bm25_retriever = (
            BM25Retriever.from_documents(
                bm25_documents
            )
        )

        self.bm25_retriever.k = BM25_K

        # ==========================================
        # Hybrid Retrieval
        # ==========================================

        self.ensemble_retriever = (
            EnsembleRetriever(
                retrievers=[
                    self.dense_retriever,
                    self.bm25_retriever
                ],
                weights=[
                    DENSE_WEIGHT,
                    BM25_WEIGHT
                ]
            )
        )

        # ==========================================
        # Cross Encoder
        # ==========================================

        self.cross_encoder = CrossEncoder(
            CROSS_ENCODER_MODEL
        )

        print("RAG retriever initialized.")

    # ==========================================
    # Retrieve + Rerank
    # ==========================================

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_K
    ):

        # ------------------------------------------
        # Hybrid retrieval
        # ------------------------------------------

        docs = self.ensemble_retriever.invoke(
            query
        )

        if not docs:
            return []

        # ------------------------------------------
        # Query-document pairs
        # ------------------------------------------

        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        # ------------------------------------------
        # Cross encoder scoring
        # ------------------------------------------

        scores = self.cross_encoder.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False
        )

        # ------------------------------------------
        # Attach scores
        # ------------------------------------------

        scored_docs = list(
            zip(docs, scores)
        )

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # ------------------------------------------
        # Final results
        # ------------------------------------------

        results = []

        for doc, score in scored_docs[:top_k]:

            results.append(
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "technology": doc.metadata.get(
                        "technology"
                    ),
                    "title": doc.metadata.get(
                        "title"
                    ),
                    "source_url": doc.metadata.get(
                        "source_url"
                    ),
                    "document": doc,
                }
            )

        return results
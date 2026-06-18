from backend.src.retrieval.retriever import Retriever
from backend.src.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:

    def __init__(self):

        self.retriever = Retriever()

    def retrieve(
        self,
        query,
        store,
        chunks
    ):

        faiss_results = self.retriever.search(
            query,
            store,
            chunks
        )

        bm25 = BM25Retriever(
            chunks
        )

        bm25_results = bm25.search(
            query
        )

        combined = (
            faiss_results +
            bm25_results
        )

        unique_results = []

        seen = set()

        for result in combined:

            key = (
                result["source"],
                result["chunk_id"]
            )

            if key not in seen:

                seen.add(key)

                unique_results.append(
                    result
                )

        return unique_results[:5]
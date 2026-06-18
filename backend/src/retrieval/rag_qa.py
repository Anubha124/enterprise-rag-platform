from backend.src.retrieval.hybrid_retriever import HybridRetriever
from backend.src.retrieval.reranker import Reranker
from backend.src.retrieval.index_manager import IndexManager
from backend.src.llm.groq_client import GroqClient


class RAGQA:

    def __init__(self):

        self.hybrid = HybridRetriever()
        self.reranker = Reranker()
        self.llm = GroqClient()

    def ask(self, file_path, question):

        # Load saved FAISS index and chunks
        store, chunks = IndexManager.load_index()

        # Hybrid Retrieval (FAISS + BM25)
        retrieved_chunks = self.hybrid.retrieve(
            question,
            store,
            chunks
        )

        # Cross Encoder Reranking
        retrieved_chunks = self.reranker.rerank(
            question,
            retrieved_chunks,
            top_k=5
        )

        # Build context
        context = "\n\n".join(
            [chunk["text"] for chunk in retrieved_chunks]
        )

        # Generate answer
        answer = self.llm.generate_answer(
            question,
            context
        )

        return answer, retrieved_chunks


if __name__ == "__main__":

    rag = RAGQA()

    answer, sources = rag.ask(
        "data/resume.pdf",
        "What projects has Anubha built?"
    )

    print("\nAnswer:\n")
    print(answer)

    print("\n📚 Sources:\n")

    for source in sources:

        print(
            f"📄 {source['source']} | Chunk {source['chunk_id']}"
        )
from backend.src.retrieval.index_manager import IndexManager
from backend.src.ingestion.document_loader import DocumentLoader
from backend.src.ingestion.text_chunker import TextChunker
from backend.src.embeddings.embedder import Embedder
from backend.src.retrieval.vector_store import VectorStore


class Retriever:

    def __init__(self):

        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedder = Embedder()

    def build_index(self, file_path):

        document = self.loader.load_document(file_path)

        raw_chunks = self.chunker.chunk_text(
            document["text"]
        )

        chunks = []

        for i, chunk in enumerate(raw_chunks):

            chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "chunk_id": i
                }
            )

        embeddings = self.embedder.create_embeddings(
            [chunk["text"] for chunk in chunks]
        )

        dimension = len(embeddings[0])

        store = VectorStore(dimension)

        store.add_embeddings(embeddings)
        IndexManager.save_index(
            store,
            chunks
            )
        return store, chunks

    def search(self, query, store, chunks):

        query_embedding = self.embedder.create_embeddings(
            [query]
        )[0]

        distances, indices = store.search(
            query_embedding,
            k=5
        )

        results = []

        for idx in indices[0]:

            results.append(
                {
                    "text": chunks[idx]["text"],
                    "source": chunks[idx]["source"],
                    "chunk_id": chunks[idx]["chunk_id"]
                }
            )

        return results


if __name__ == "__main__":

    retriever = Retriever()

    store, chunks = retriever.build_index(
        "data/resume.pdf"
    )

    query = "What projects has Anubha built?"

    results = retriever.search(
        query,
        store,
        chunks
    )

    print("\nRelevant Chunks:\n")

    for i, result in enumerate(results, start=1):

        print(f"\nChunk {i}\n")

        print(result["text"][:500])

        print(
            f"\nSource: {result['source']} | Chunk ID: {result['chunk_id']}"
        )
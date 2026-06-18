from pathlib import Path

from backend.src.ingestion.document_loader import DocumentLoader
from backend.src.ingestion.text_chunker import TextChunker
from backend.src.embeddings.embedder import Embedder
from backend.src.retrieval.vector_store import VectorStore


class RAGPipeline:

    def __init__(self):

        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedder = Embedder()

    def process_document(self, file_path):

        document = self.loader.load_document(file_path)

        chunks = self.chunker.chunk_text(
            document["text"]
        )

        embeddings = self.embedder.create_embeddings(
            chunks
        )

        dimension = len(embeddings[0])

        store = VectorStore(dimension)

        store.add_embeddings(embeddings)

        return store, chunks


if __name__ == "__main__":

    pipeline = RAGPipeline()

    store, chunks = pipeline.process_document(
        "data/resume.pdf"
    )

    print(f"Total Chunks: {len(chunks)}")

    print("\nFirst Chunk:\n")

    print(chunks[0][:500])
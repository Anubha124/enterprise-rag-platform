from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):

        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

    def create_embeddings(self, chunks):

        embeddings = self.model.encode(chunks)

        return embeddings


if __name__ == "__main__":

    chunks = [
        "Artificial Intelligence is transforming software engineering.",
        "RAG systems combine retrieval and generation.",
        "Machine Learning uses data to make predictions."
    ]

    embedder = Embedder()

    embeddings = embedder.create_embeddings(chunks)

    print(f"Number of Embeddings: {len(embeddings)}")

    print(f"Embedding Dimension: {len(embeddings[0])}")
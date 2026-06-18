import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):

        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings):

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        self.index.add(embeddings)

    def search(self, query_embedding, k=3):

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        return distances, indices


if __name__ == "__main__":

    embeddings = np.random.rand(
        5,
        384
    ).astype("float32")

    store = VectorStore(384)

    store.add_embeddings(embeddings)

    query = np.random.rand(
        384
    ).astype("float32")

    distances, indices = store.search(query)

    print("Distances:")
    print(distances)

    print("\nIndices:")
    print(indices)
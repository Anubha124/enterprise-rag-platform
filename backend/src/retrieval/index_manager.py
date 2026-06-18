import joblib
import faiss

from backend.src.retrieval.vector_store import VectorStore


class IndexManager:

    @staticmethod
    def save_index(store, chunks):

        faiss.write_index(
            store.index,
            "vectorstore/faiss.index"
        )

        joblib.dump(
            chunks,
            "vectorstore/chunks.pkl"
        )

    @staticmethod
    def load_index():

        index = faiss.read_index(
            "vectorstore/faiss.index"
        )

        chunks = joblib.load(
            "vectorstore/chunks.pkl"
        )

        dimension = index.d

        store = VectorStore(dimension)

        store.index = index

        return store, chunks
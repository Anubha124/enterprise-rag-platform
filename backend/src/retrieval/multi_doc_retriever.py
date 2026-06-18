from backend.src.retrieval.retriever import Retriever


class MultiDocRetriever:

    def __init__(self):

        self.retriever = Retriever()

    def build_indexes(
        self,
        file_paths
    ):

        all_chunks = []

        stores = []

        for path in file_paths:

            store, chunks = self.retriever.build_index(
                path
            )

            stores.append(store)

            all_chunks.extend(chunks)

        return stores, all_chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

    def chunk_text(self, text):

        chunks = self.splitter.split_text(text)

        return chunks


if __name__ == "__main__":

    sample_text = """
    Artificial Intelligence is transforming software engineering.

    RAG systems combine retrieval and generation.

    Large Language Models can answer questions based on external documents.

    This is a sample text for testing chunking.
    """ * 20

    chunker = TextChunker()

    chunks = chunker.chunk_text(sample_text)

    print(f"Total Chunks: {len(chunks)}")

    print("\nFirst Chunk:\n")

    print(chunks[0])
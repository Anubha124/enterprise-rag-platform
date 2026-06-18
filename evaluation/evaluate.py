import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)
from test_questions import test_questions
from backend.src.retrieval.index_manager import IndexManager
from backend.src.retrieval.hybrid_retriever import HybridRetriever


store, chunks = IndexManager.load_index()

retriever = HybridRetriever()

correct = 0
total = len(test_questions)

for test in test_questions:

    results = retriever.retrieve(
        test["question"],
        store,
        chunks
    )

    retrieved_chunk_ids = [
        result["chunk_id"]
        for result in results
    ]

    if test["expected_chunk"] in retrieved_chunk_ids:

        correct += 1

    print("\nQuestion:")
    print(test["question"])

    print("Expected Chunk:")
    print(test["expected_chunk"])

    print("Retrieved:")
    print(retrieved_chunk_ids)

accuracy = (correct / total) * 100

print("\n====================")
print(f"Hit Rate@5: {accuracy:.2f}%")
print("====================")
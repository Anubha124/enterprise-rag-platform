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

total_rr = 0

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

    rank = None

    for i, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1
    ):

        if chunk_id == test["expected_chunk"]:

            rank = i
            break

    if rank:

        rr = 1 / rank

    else:

        rr = 0

    total_rr += rr

    print("\nQuestion:")
    print(test["question"])

    print("Expected Chunk:")
    print(test["expected_chunk"])

    print("Retrieved:")
    print(retrieved_chunk_ids)

    print("Rank:")
    print(rank)

    print("Reciprocal Rank:")
    print(rr)

mrr = total_rr / len(test_questions)

print("\n====================")
print(f"MRR: {mrr:.4f}")
print("====================")
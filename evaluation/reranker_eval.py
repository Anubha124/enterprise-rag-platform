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
from backend.src.retrieval.reranker import Reranker


store, chunks = IndexManager.load_index()

hybrid = HybridRetriever()
reranker = Reranker()


def calculate_mrr(results_list):

    total_rr = 0

    for test in test_questions:

        results = results_list[test["question"]]

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

            total_rr += (1 / rank)

    return total_rr / len(test_questions)


hybrid_results = {}
reranked_results = {}

for test in test_questions:

    results = hybrid.retrieve(
        test["question"],
        store,
        chunks
    )

    hybrid_results[
        test["question"]
    ] = results

    reranked = reranker.rerank(
        test["question"],
        results
    )

    reranked_results[
        test["question"]
    ] = reranked


hybrid_mrr = calculate_mrr(
    hybrid_results
)

reranked_mrr = calculate_mrr(
    reranked_results
)

print("\n====================")
print(f"Hybrid MRR: {hybrid_mrr:.4f}")
print(f"Reranked MRR: {reranked_mrr:.4f}")
print("====================")
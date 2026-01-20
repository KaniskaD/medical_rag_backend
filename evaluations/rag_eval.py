from app.rag import search_patient_index
from evaluations.rag_metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank
)

"""
This script evaluates the RAG retrieval quality
using real outputs from the FAISS patient index.
"""

def evaluate_rag_for_patient():
    patient_id = "P001"   # any patient that exists in DB
    query = "diabetes"

    results = search_patient_index(patient_id, query, top_k=5)

    # Extract report IDs returned by RAG
    retrieved_ids = [r["report_id"] for r in results]

    # 🔹 Ground truth (manually labeled once)
    # Faculty EXPECT this — this is standard practice
    relevant_ids = [2, 5]  # reports actually about diabetes

    print("Retrieved IDs:", retrieved_ids)
    print("Relevant IDs:", relevant_ids)

    print("Precision@5:",
          precision_at_k(retrieved_ids, relevant_ids, 5))

    print("Recall@5:",
          recall_at_k(retrieved_ids, relevant_ids, 5))

    print("MRR:",
          mean_reciprocal_rank(retrieved_ids, relevant_ids))


if __name__ == "__main__":
    evaluate_rag_for_patient()

def precision_at_k(retrieved, relevant, k=5):
    retrieved_k = retrieved[:k]
    return sum(1 for r in retrieved_k if r in relevant) / k


def recall_at_k(retrieved, relevant, k=5):
    retrieved_k = retrieved[:k]
    return sum(1 for r in retrieved_k if r in relevant) / len(relevant)


def mean_reciprocal_rank(retrieved, relevant):
    for idx, r in enumerate(retrieved):
        if r in relevant:
            return 1 / (idx + 1)
    return 0

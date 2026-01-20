import re

def normalize(text: str):
    return set(
        re.sub(r"[^a-zA-Z0-9 ]", "", text.lower()).split()
    )

def faithfulness_score(answer: str, retrieved_context: str) -> float:
    """
    Measures how much of the generated answer is grounded
    in the retrieved documents.
    """
    if not answer or not retrieved_context:
        return 0.0

    answer_tokens = normalize(answer)
    context_tokens = normalize(retrieved_context)

    if not answer_tokens:
        return 0.0

    supported_tokens = answer_tokens.intersection(context_tokens)

    return len(supported_tokens) / len(answer_tokens)

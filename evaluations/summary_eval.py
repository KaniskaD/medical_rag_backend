from app.llm import generate_text
from evaluations.summarization_metrics import compute_rouge

"""
Evaluates summarization quality using ROUGE-L
between system-generated and reference summaries.
"""

def evaluate_summary():
    report_text = """
    name: aaRon MARtiNeZ
    age: 38
    gender: Female
    blood type: A-
    health condition: Hypertension
    admission type: Urgent
    medication: Lipitor
    test results: Inconclusive
    """

    generated_summary = generate_text(
        system_prompt="Generate a concise medical summary",
        user_prompt=report_text,
        max_tokens=150
    )

    reference_summary = (
        """The patient presents with Hypertension, a chronic condition characterized by elevated blood pressure.
        She is currently managed with Lipitor (Atorvastatin). While Lipitor is primarily a lipid-lowering medication used to
        treat high cholesterol, it is frequently prescribed to patients with hypertension to reduce the overall risk of cardiovascular
        events like heart attack or stroke.
        The "Urgent" admission status, combined with an "Inconclusive" test result, suggests that further diagnostic screening is likely 
        required to determine the acute cause of her visit. Her blood type (A-) is relatively rare (found in approximately 6% of 
        the population), which is a critical data point should she require a transfusion during her stay.
        """
    )

    rouge_l = compute_rouge(reference_summary, generated_summary)

    print("Generated Summary:\n", generated_summary)
    print("\nReference Summary:\n", reference_summary)
    print("\nROUGE-L Score:", rouge_l)


if __name__ == "__main__":
    evaluate_summary()

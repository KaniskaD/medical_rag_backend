from app.rag import search_patient_index
from evaluations.faithfulness_metric import faithfulness_score

patient_id = "P001"
query = "diabetes condition"

results = search_patient_index(patient_id, query, top_k=3)

retrieved_text = " ".join(
    r.get("text", "") for r in results
)

generated_answer = """
Hello, I'm your medical assistant. I'd like to take a few minutes to review your health record with you.
*Your Personal Information* My name is aaRon MARtiNeZ, and I'm 38 years old. My gender is female, and my blood type is A-.
*Health Condition* You have been diagnosed with hypertension, which means your blood pressure is higher than normal.
This can increase your risk of developing other health problems if it's not managed properly.
*Recent Visit* You were admitted to the hospital urgently on December 25th, 2025.
Unfortunately, some of your test results are inconclusive, which means we need to run more tests to get a clearer picture of what's
going on with your body. *Medications* You're currently taking Lipitor, which is a medication that helps manage your blood pressure.
I want to reassure you that this medication is helping to keep your blood pressure under control. 
That's the main information from your health record. If you have any questions or concerns, please don't hesitate to ask me. 
We'll work together to make sure you get the best possible care.
"""

score = faithfulness_score(generated_answer, retrieved_text)

print("Faithfulness Score:", round(score, 3))

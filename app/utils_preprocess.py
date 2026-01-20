import re
import unicodedata


def clean_medical_text(text: str) -> str:
    """
    Perform healthcare-specific preprocessing on extracted text
    BEFORE generating embeddings for RAG.
    """

    if not text or not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = text.replace("•", "- ").replace("", "- ")

    text = re.sub(r"[^\x20-\x7E\n\t]", "", text)

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)


    text = re.sub(
        r"([A-Za-z][A-Za-z0-9 %/()-]{3,})\n([0-9].{0,12})",
        r"\1: \2",
        text
    )

    unit_map = {
        "mg/dl": "mg/dL",
        "mg / dl": "mg/dL",
        "mmol/l": "mmol/L",
        "mmol / l": "mmol/L",
        "iu/l": "IU/L",
        "iu / l": "IU/L",
        "g/dl": "g/dL",
        "g / dl": "g/dL"
    }

    for k, v in unit_map.items():
        text = re.sub(k, v, text, flags=re.IGNORECASE)

    text = re.sub(r"(\d)O(\d)", r"\1 0 \2", text)   # 1O4 -> 1 0 4
    text = re.sub(r"O(?=\d)", "0", text)            # O5 -> 05
    text = re.sub(r"(?<=\d)O", "0", text)           # 50 -> 50

    abbrev = {
        r"\bBP\b": "Blood Pressure",
        r"\bHR\b": "Heart Rate",
        r"\bRR\b": "Respiratory Rate",
        r"\bTemp\b": "Temperature",
        r"\bDx\b": "Diagnosis",
        r"\bRx\b": "Prescription",
        r"\bHx\b": "History",
        r"\bTx\b": "Treatment",
    }

    for pattern, expanded in abbrev.items():
        text = re.sub(pattern, expanded, text)

    lines = text.split("\n")
    cleaned_lines = []
    seen = set()

    for line in lines:
        stripped = line.strip()
        if stripped not in seen:
            cleaned_lines.append(line)
            seen.add(stripped)

    text = "\n".join(cleaned_lines)

    text = text.strip()

    return text

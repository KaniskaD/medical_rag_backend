from typing import Dict
from collections import defaultdict
from app.utils_analytics import safe_dict

# -------------------------
# LAB ANALYTICS
# -------------------------
def lab_analytics(patient_data: Dict):
    reports = patient_data.get("reports", [])
    trends = defaultdict(list)

    for r in reports:
        if r.report_type == "lab":
            extracted = safe_dict(r.extracted_data)
            for k, v in extracted.items():
                trends[k].append(v)

    return {
        "type": "lab",
        "trends": dict(trends),
    }


# -------------------------
# IMAGE ANALYTICS (stub)
# -------------------------
def image_analytics(patient_data: Dict):
    return {
        "type": "image",
        "message": "Imaging analytics available",
    }


# -------------------------
# AUDIO ANALYTICS (stub)
# -------------------------
def audio_analytics(patient_data: Dict):
    return {
        "type": "audio",
        "message": "Audio symptom analysis available",
    }

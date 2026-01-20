import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..deps import get_db
from .auth import get_current_user
from ..llm import generate_text

router = APIRouter(prefix="/summary", tags=["summary"])


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _build_lab_context(reports):
    blocks = []
    for r in reports:
        if r.report_type == "lab" and r.extracted_data:
            ts = r.created_at.strftime("%Y-%m-%d")
            blocks.append(
                f"[LAB REPORT | {ts}]\n" +
                json.dumps(r.extracted_data, indent=2)
            )
    return "\n\n---\n\n".join(blocks)

def _lab_report_to_text(report: models.Report) -> str:
    """
    Converts structured lab data into readable text for LLMs.
    """
    if not report.extracted_data:
        return ""

    lines = [
        f"{key}: {value}"
        for key, value in report.extracted_data.items()
    ]

    return (
        f"[LAB REPORT | {report.created_at.strftime('%Y-%m-%d')}]\n"
        + "\n".join(lines)
    )

def _build_context_from_reports(reports):
    texts = []

    for r in reports:
        timestamp = r.created_at.strftime("%Y-%m-%d")

        # 🔹 TEXT / IMAGE REPORTS
        if r.report_type in ("text", "image") and r.parsed_text:
            texts.append(
                f"[REPORT | {r.report_type.upper()} | {timestamp}]\n{r.parsed_text}"
            )

        # 🔹 LAB REPORTS (NEW)
        elif r.report_type == "lab":
            lab_text = _lab_report_to_text(r)
            if lab_text:
                texts.append(lab_text)

    if not texts:
        return ""

    full = "\n\n---\n\n".join(texts)

    if len(full) > 8000:
        full = full[-8000:]

    return full


def _generate_patient_summary(patient_id: str, context: str):
    system_prompt = (
        "You are a medical assistant that explains a patient's health record "
        "in simple, clear, and reassuring language. Avoid medical jargon where possible."
    )

    user_prompt = f"""
    Patient ID: {patient_id}

    Patient records:
    {context}

    Task:
    - Summarize this information in simple language for the patient.
    - Use short paragraphs or bullet points.
    - Do not invent facts.
    - Do NOT prescribe medications.
    """

    return generate_text(system_prompt, user_prompt, max_tokens=400)


def _generate_doctor_summary(patient_id: str, context: str):
    system_prompt = (
        "You are a clinical decision-support assistant generating concise, "
        "medically accurate summaries strictly from patient records."
    )

    user_prompt = f"""
Patient ID: {patient_id}

Patient records:
{context}

Task:
- Provide a concise clinical summary.
- Highlight abnormal findings and trends.
- Do not speculate beyond the text.
"""

    return generate_text(system_prompt, user_prompt, max_tokens=400)


# --------------------------------------------------
# Patient summary
# --------------------------------------------------
@router.get("/{patient_id}/patient", response_model=schemas.SummaryResponse)
def get_patient_friendly_summary(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if current_user.role == "patient" and current_user.patient_id != patient_id:
        raise HTTPException(status_code=403, detail="Not allowed")

    if current_user.role == "lab_tech":
        raise HTTPException(status_code=403, detail="Lab technicians cannot request summaries")

    patient = db.query(models.Patient).filter(
        models.Patient.patient_id == patient_id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    reports = (
        db.query(models.Report)
        .filter(models.Report.patient_id == patient_id)
        .order_by(models.Report.created_at.asc())
        .all()
    )

    context = _build_context_from_reports(reports)

    context = _build_context_from_reports(reports)
    lab_context = _build_lab_context(reports)
    full_context = "\n\n".join([context, lab_context]).strip()

    if not context:
        summary_text = (
            "No textual medical reports are available yet. "
            "Your lab results are recorded, but a written summary requires "
            "doctor notes or diagnostic reports."
        )
    else:
        summary_text = _generate_patient_summary(patient_id, context)

    return schemas.SummaryResponse(
        patient_id=patient_id,
        summary_type="patient",
        summary=summary_text,
    )


# --------------------------------------------------
# Doctor summary
# --------------------------------------------------
@router.get("/{patient_id}/doctor", response_model=schemas.SummaryResponse)
def get_doctor_friendly_summary(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if current_user.role == "lab_tech":
        raise HTTPException(status_code=403, detail="Lab technicians cannot request summaries")

    patient = db.query(models.Patient).filter(
        models.Patient.patient_id == patient_id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    reports = (
        db.query(models.Report)
        .filter(models.Report.patient_id == patient_id)
        .order_by(models.Report.created_at.asc())
        .all()
    )

    context = _build_context_from_reports(reports)
    context = _build_context_from_reports(reports)
    lab_context = _build_lab_context(reports)

    full_context = "\n\n".join([context, lab_context]).strip()

    if not context:
        summary_text = "No textual reports available yet for this patient."
    else:
        summary_text = _generate_doctor_summary(patient_id, context)

    return schemas.SummaryResponse(
        patient_id=patient_id,
        summary_type="doctor",
        summary=summary_text,
    )
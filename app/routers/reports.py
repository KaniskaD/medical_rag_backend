import os
import csv
import io
import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from .. import models, schemas
from ..deps import get_db
from .auth import get_current_user

from ..utils_text import extract_text_from_file, calculate_content_hash
from ..utils_preprocess import clean_medical_text
from ..utils_image import extract_image_embedding, generate_image_caption

from ..rag import (
    add_text_to_index,
    add_image_to_index,
    search_patient_index,
    search_by_vector,
    get_patient_index_stats
)

from ..llm import generate_text

router = APIRouter(prefix="/reports", tags=["reports"])

REPORTS_DIR = os.path.join("storage", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# =========================================================
# SINGLE REPORT UPLOAD
# Supports:
# - text/image files
# - structured lab JSON (no file)
# - duplicate prevention via content_hash
# =========================================================
@router.post("/upload", response_model=schemas.ReportOut)
async def upload_report(
    patient_id: str = Form(...),
    report_type: str = Form("text"),
    extracted_data: str | None = Form(None),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    # ---------------------------------------------------------
    # 0) Ensure patient exists
    # ---------------------------------------------------------
    patient = (
        db.query(models.Patient)
        .filter(models.Patient.patient_id == patient_id)
        .first()
    )
    if not patient:
        patient = models.Patient(
            patient_id=patient_id,
            name=f"Patient {patient_id}",
            dob="01/01/2000",
            gender="Unknown"
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

    # ---------------------------------------------------------
    # 1) LAB JSON (no file)
    # ---------------------------------------------------------
    if report_type == "lab":
        try:
            parsed = json.loads(extracted_data or "{}")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid lab JSON")

        db_report = models.Report(
            patient_id=patient_id,
            uploaded_by=current_user.id,
            uploader_role=current_user.role,
            report_type="lab",
            extracted_data=parsed,
            parsed_text=f"Structured lab report\n\n{json.dumps(parsed, indent=2)}",
            file_path=f"lab://{patient_id}/{datetime.utcnow().isoformat()}",
            created_at=datetime.utcnow(),
        )

        db.add(db_report)
        db.commit()
        db.refresh(db_report)

        add_text_to_index(patient_id, db_report.id, db_report.parsed_text)
        return db_report

    # ---------------------------------------------------------
    # 2) FILE REQUIRED FOR NON-LAB
    # ---------------------------------------------------------
    if not file:
        raise HTTPException(
            status_code=400,
            detail="File required for text/image reports"
        )

    file_bytes = await file.read()
    content_hash = calculate_content_hash(file_bytes)

    # Duplicate prevention
    duplicate = (
        db.query(models.Report)
        .filter(
            models.Report.patient_id == patient_id,
            models.Report.content_hash == content_hash
        )
        .first()
    )
    if duplicate:
        raise HTTPException(
            status_code=400,
            detail="Duplicate report detected for this patient"
        )

    ext = os.path.splitext(file.filename)[1].lower()
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{patient_id}_{timestamp}{ext}"

    patient_dir = os.path.join(REPORTS_DIR, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    file_path = os.path.join(patient_dir, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    raw_text = extract_text_from_file(file_path)

    # ---------------------------------------------------------
    # 3) Auto-detect lab files (.json / .csv)
    # ---------------------------------------------------------
    if ext in [".json", ".csv"]:
        parsed_text = f"Structured lab report\n\n{raw_text}"

        db_report = models.Report(
            patient_id=patient_id,
            uploaded_by=current_user.id,
            uploader_role=current_user.role,
            report_type="lab",
            extracted_data=None,
            parsed_text=parsed_text,
            file_path=file_path,
            content_hash=content_hash,
            created_at=datetime.utcnow(),
        )

        db.add(db_report)
        db.commit()
        db.refresh(db_report)

        add_text_to_index(patient_id, db_report.id, parsed_text)
        return db_report

    # ---------------------------------------------------------
    # 4) TEXT / IMAGE REPORT
    # ---------------------------------------------------------
    parsed_text = clean_medical_text(raw_text)
    is_image = ext in [".png", ".jpg", ".jpeg"]

    db_report = models.Report(
        patient_id=patient_id,
        uploaded_by=current_user.id,
        uploader_role=current_user.role,
        file_path=file_path,
        content_hash=content_hash,
        parsed_text=parsed_text,
        report_type="image" if is_image else "text",
        created_at=datetime.utcnow(),
    )

    db.add(db_report)
    db.commit()
    db.refresh(db_report)

    # ---------------------------------------------------------
    # 5) IMAGE: embeddings + RAG + LLM summaries
    # ---------------------------------------------------------
    if is_image:
        embedding = extract_image_embedding(file_path)
        add_image_to_index(patient_id, db_report.id, embedding)

        related = search_by_vector(patient_id, embedding, top_k=5)
        retrieved_text = "\n\n".join(
            r["text"] for r in related if r.get("type") == "text"
        )

        caption = generate_image_caption(file_path)

        doctor_prompt = (
            "You are a radiologist. Generate a concise clinical impression."
        )
        doctor_input = f"{caption}\n\n{parsed_text}\n\n{retrieved_text}"

        patient_prompt = (
            "Explain the findings in simple language for a patient."
        )
        patient_input = doctor_input

        doctor_summary = generate_text(doctor_prompt, doctor_input, max_tokens=300)
        patient_summary = generate_text(patient_prompt, patient_input, max_tokens=200)

        parsed_text += (
            f"\n\n[DOCTOR SUMMARY]\n{doctor_summary}\n\n"
            f"[PATIENT SUMMARY]\n{patient_summary}"
        )

        db_report.parsed_text = parsed_text
        db.commit()

        add_text_to_index(patient_id, db_report.id, parsed_text)

    else:
        if parsed_text:
            add_text_to_index(patient_id, db_report.id, parsed_text)

    return db_report


# =========================================================
# CSV BULK LAB UPLOAD
# =========================================================
@router.post("/upload-csv")
async def upload_lab_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if current_user.role not in ["lab_tech", "lab technician", "admin"]:
        raise HTTPException(status_code=403, detail="Lab access required")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    decoded = (await file.read()).decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))

    created = 0
    for row in reader:
        patient_id = row.get("patient_id")
        if not patient_id:
            continue

        extracted_data = {
            k: float(v)
            for k, v in row.items()
            if k != "patient_id" and v not in ("", None)
        }

        report = models.Report(
            patient_id=patient_id,
            uploaded_by=current_user.id,
            uploader_role=current_user.role,
            report_type="lab",
            extracted_data=extracted_data,
            parsed_text=json.dumps(extracted_data, indent=2),
            file_path=f"labcsv://{patient_id}/{datetime.utcnow().isoformat()}",
            created_at=datetime.utcnow(),
        )

        db.add(report)
        db.commit()
        db.refresh(report)

        add_text_to_index(patient_id, report.id, report.parsed_text)
        created += 1

    return {
        "message": "CSV lab upload completed",
        "reports_created": created
    }


# =========================================================
# EXISTING ENDPOINTS
# =========================================================
@router.get("/by-patient/{patient_id}", response_model=List[schemas.ReportOut])
def get_reports_for_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return (
        db.query(models.Report)
        .filter(models.Report.patient_id == patient_id)
        .order_by(models.Report.created_at.desc())
        .all()
    )


@router.get("/search/{patient_id}")
def search_reports_for_patient(
    patient_id: str,
    query: str,
    top_k: int = 5,
    current_user: models.User = Depends(get_current_user),
):
    results = search_patient_index(patient_id, query, top_k)
    return {"patient_id": patient_id, "query": query, "results": results}


@router.get("/debug-index/{patient_id}")
def debug_patient_index(
    patient_id: str,
    current_user: models.User = Depends(get_current_user),
):
    return get_patient_index_stats(patient_id)


@router.delete("/{report_id}")
def delete_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can delete reports")

    report = db.query(models.Report).filter(
        models.Report.id == report_id
    ).first()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    db.delete(report)
    db.commit()
    return {"message": "Report deleted successfully"}

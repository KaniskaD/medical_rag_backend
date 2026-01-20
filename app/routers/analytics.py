from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from collections import defaultdict

from app.utils_analytics import safe_dict, safe_list
from app.deps import get_db
from app import models
from app.routers.auth import get_current_user

from app.analytics.registry import ANALYTICS_REGISTRY
import app.analytics.register  # ensures registry is populated

router = APIRouter(prefix="/analytics", tags=["analytics"])

# --------------------------------------------------
# Utility helpers (SAFE)
# --------------------------------------------------

def detect_modalities(reports):
    modalities = set()
    for r in reports:
        if r.report_type:
            modalities.add(r.report_type)
    return modalities


def compute_lab_trends(reports):
    trends = defaultdict(list)
    for r in reports:
        if r.report_type == "lab":
            extracted = safe_dict(r.extracted_data)
            for k, v in extracted.items():
                trends[k].append(v)
    return dict(trends)


def compute_simple_risk(trends):
    risk = "low"

    hba1c_vals = safe_list(trends.get("hba1c"))
    if hba1c_vals:
        try:
            max_val = max(hba1c_vals)
            if max_val >= 8:
                risk = "high"
            elif max_val >= 6.5:
                risk = "medium"
        except Exception:
            pass

    return risk


def compute_risk_distribution(db: Session):
    """
    Population-level risk distribution across all patients
    """
    patients = db.query(models.Patient).all()
    dist = {"low": 0, "medium": 0, "high": 0}

    for p in patients:
        reports = (
            db.query(models.Report)
            .filter(models.Report.patient_id == p.patient_id)
            .all()
        )

        lab_trends = compute_lab_trends(reports)
        risk = compute_simple_risk(lab_trends)
        dist[risk] += 1

    return dist

def run_registry_analytics(modalities, reports):
    results = {}
    patient_data = {
        "reports": reports,
        "modalities": modalities,
    }

    for name, entry in ANALYTICS_REGISTRY.items():
        if any(req in modalities for req in entry["requires"]):
            results[name] = entry["func"](patient_data)

    return results

# --------------------------------------------------
# PATIENT ANALYTICS
# --------------------------------------------------
@router.get("/patient/{patient_id}")
def patient_analytics(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    if current_user.role == "patient" and current_user.patient_id != patient_id:
        raise HTTPException(status_code=403, detail="Access denied")

    reports = (
        db.query(models.Report)
        .filter(models.Report.patient_id == patient_id)
        .all()
    )

    if not reports:
        return {
            "patient_id": patient_id,
            "available_data": [],
            "lab_trends": {},
            "risk_level": "unknown",
            "total_reports": 0,
            "report_distribution": {},
            "message": "No reports uploaded yet",
        }

    modalities = detect_modalities(reports)
    lab_trends = compute_lab_trends(reports) if "lab" in modalities else {}
    adaptive_analytics = run_registry_analytics(modalities, reports)

    report_distribution = defaultdict(int)
    for r in reports:
        if r.report_type:
            report_distribution[r.report_type] += 1

    return {
        "patient_id": patient_id,
        "available_data": list(modalities),
        "lab_trends": lab_trends,
        "risk_level": compute_simple_risk(lab_trends),
        "total_reports": len(reports),
        "adaptive_analytics": adaptive_analytics,
        "report_distribution": dict(report_distribution),
        "message": "Analytics shown only for available data",
    }


# --------------------------------------------------
# DOCTOR ANALYTICS (PER PATIENT)
# --------------------------------------------------
@router.get("/doctor/{patient_id}")
def doctor_analytics_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Doctor access required")

    reports = (
        db.query(models.Report)
        .filter(models.Report.patient_id == patient_id)
        .all()
    )

    if not reports:
        return {
            "patient_id": patient_id,
            "modalities_detected": [],
            "lab_trends": {},
            "risk_level": "unknown",
            "clinical_notes": "No reports available for this patient",
        }

    modalities = detect_modalities(reports)
    lab_trends = compute_lab_trends(reports) if "lab" in modalities else {}

    return {
        "patient_id": patient_id,
        "modalities_detected": list(modalities),
        "lab_trends": lab_trends,
        "risk_level": compute_simple_risk(lab_trends),
        "clinical_notes": "Use trends and risk for decision making",
    }


# --------------------------------------------------
# DOCTOR ANALYTICS (POPULATION LEVEL)
# --------------------------------------------------
@router.get("/doctor")
def doctor_analytics_population(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Doctor access required")

    risk_dist = compute_risk_distribution(db)

    return {
        "total_patients": db.query(models.Patient).count(),
        "reports_today": db.query(models.Report).count(),
        "risk_distribution": [
            {"name": "Low Risk", "value": risk_dist["low"]},
            {"name": "Medium Risk", "value": risk_dist["medium"]},
            {"name": "High Risk", "value": risk_dist["high"]},
        ],
    }
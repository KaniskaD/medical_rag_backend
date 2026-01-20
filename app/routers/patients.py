from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..deps import get_db

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("/", response_model=schemas.PatientOut)
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    existing = db.query(models.Patient).filter(models.Patient.patient_id == patient.patient_id).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Patient ID already exists")

    db_patient = models.Patient(
        patient_id=patient.patient_id,
        name=patient.name,
        dob=patient.dob,
        gender=patient.gender,
        primary_doctor_id=patient.primary_doctor_id,
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


@router.get("/", response_model=List[schemas.PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(models.Patient).all()


@router.get("/{patient_id}", response_model=schemas.PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    return patient

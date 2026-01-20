from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# ---------- Auth / Token ----------

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


# ---------- User ----------

class UserBase(BaseModel):
    username: str
    role: str  # "admin" | "doctor" | "lab_tech" | "patient"


class UserCreate(UserBase):
    password: str
    patient_id: Optional[str] = None  # link for patient accounts


class UserOut(UserBase):
    id: int
    patient_id: Optional[str] = None

    class Config:
        orm_mode = True


# ---------- Patient ----------

class PatientBase(BaseModel):
    patient_id: str
    name: str
    dob: Optional[str] = None
    gender: Optional[str] = None
    primary_doctor_id: Optional[int] = None


class PatientCreate(PatientBase):
    pass


class PatientOut(PatientBase):
    id: int

    class Config:
        orm_mode = True


# ---------- Report ----------


class ReportOut(BaseModel):
    id: int
    patient_id: str
    uploader_role: str
    file_path: Optional[str] = None
    parsed_text: Optional[str]
    created_at: datetime
    extracted_data: Optional[dict] = None
    report_type: Optional[str]
    source_label: Optional[str]

    class Config:
        orm_mode = True


# ---------- LabResult ----------


class LabResultOut(BaseModel):
    id: int
    patient_id: str
    report_id: int
    test_name: str
    value: float
    unit: Optional[str]
    is_abnormal: bool
    timestamp: datetime

    class Config:
        orm_mode = True


# ---------- Chat & RAG ----------


class ChatCreate(BaseModel):
    patient_id: str
    asked_by_role: str
    question: str


class ChatOut(BaseModel):
    id: int
    patient_id: str
    asked_by_role: str
    question: str
    answer: str
    created_at: datetime

    class Config:
        orm_mode = True


class ChatRequest(BaseModel):
    question: str


class SummaryResponse(BaseModel):
    patient_id: str
    summary_type: str  # "doctor" or "patient"
    summary: str

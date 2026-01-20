from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Float,
    Boolean,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy import JSON
from datetime import datetime

from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # admin | doctor | lab_tech | patient

    # For patient users
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=True)

    # Doctor → Patients relationship
    doctor_patients = relationship(
        "Patient",
        back_populates="primary_doctor",
        foreign_keys="Patient.primary_doctor_id",
    )

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    dob = Column(String, nullable=True)
    gender = Column(String, nullable=True)

    primary_doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    primary_doctor = relationship(
        "User",
        back_populates="doctor_patients",
        foreign_keys=[primary_doctor_id],
    )

    reports = relationship("Report", back_populates="patient")
    lab_results = relationship("LabResult", back_populates="patient")
    chats = relationship("ChatHistory", back_populates="patient")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    uploader_role = Column(String, nullable=False)

    file_path = Column(Text, nullable=True)

    content_hash = Column(String, index=True, nullable=True)

    parsed_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    report_type = Column(String, nullable=True)  # text | image | lab
    source_label = Column(String, nullable=True)

    extracted_data = Column(JSON, nullable=True)

    patient = relationship("Patient", back_populates="reports")
    uploader = relationship("User")

class LabResult(Base):
    __tablename__ = "lab_results"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)

    test_name = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=True)
    is_abnormal = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="lab_results")
    report = relationship("Report")

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    asked_by_role = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="chats")

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from sqlalchemy.orm import Session
import os, uuid
from typing import Optional
from .. import models, schemas
from ..deps import get_db
from .auth import get_current_user
from ..rag import search_patient_index
from ..llm import generate_text
from ..utils_audio import transcribe_audio_file, text_to_speech

router = APIRouter(prefix="/chat", tags=["chat"])

AUDIO_RESPONSE_DIR = "storage/audio_responses"
TMP_AUDIO_DIR = "tmp_audio"
os.makedirs(AUDIO_RESPONSE_DIR, exist_ok=True)
os.makedirs(TMP_AUDIO_DIR, exist_ok=True)

def build_lab_context(db: Session, patient_id: str) -> str:
    """
    Builds readable lab context from structured lab reports.
    Used when FAISS text context is empty or insufficient.
    """
    reports = (
        db.query(models.Report)
        .filter(
            models.Report.patient_id == patient_id,
            models.Report.report_type == "lab",
        )
        .order_by(models.Report.created_at.desc())
        .all()
    )

    if not reports:
        return ""

    blocks = []
    for r in reports:
        if not r.extracted_data:
            continue

        ts = r.created_at.strftime("%Y-%m-%d")
        
        # Added handling for dict vs string for analytics module support
        if isinstance(r.extracted_data, dict):
            lines = [f"{k}: {v}" for k, v in r.extracted_data.items()]
            content = "\n".join(lines)
        else:
            content = str(r.extracted_data)

        blocks.append(
            f"[LAB REPORT | {ts}]\n" + content
        )

    return "\n\n---\n\n".join(blocks)

def safe_search_patient_index(patient_id: str, query: str, top_k: int = 8):
    """
    Wrapper to safely handle empty or missing FAISS index
    (e.g., patients with only lab reports).
    """
    try:
        results = search_patient_index(patient_id, query=query, top_k=top_k)
        if not isinstance(results, list):
            return []
        return results
    except Exception as e:
        print("RAG search skipped:", e)
        return []

@router.post("/{patient_id}", response_model=schemas.ChatOut)
async def chat_with_patient_history(
    patient_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if current_user.role == "patient" and current_user.patient_id != patient_id:
        raise HTTPException(status_code=403, detail="Not allowed")

    if current_user.role not in ["doctor", "patient"]:
        raise HTTPException(status_code=403, detail="Not allowed")

    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    content_type = request.headers.get("content-type", "") or ""
    question: Optional[str] = None
    upload_file: Optional[UploadFile] = None
    language = "English"   # DEFAULT

    if content_type.startswith("multipart/form-data"):
        form = await request.form()

        q_val = form.get("question")
        if isinstance(q_val, str):
            question = q_val.strip() or None

        lang_val = form.get("language")
        if isinstance(lang_val, str):
            language = lang_val

        f_val = form.get("file")
        if isinstance(f_val, UploadFile):
            upload_file = f_val

    elif content_type.startswith("application/json"):
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        q_val = data.get("question")
        if isinstance(q_val, str):
            question = q_val.strip() or None

        language = data.get("language", "English")

    else:
        raise HTTPException(status_code=415, detail="Unsupported Content-Type")

    if not question and not upload_file:
        raise HTTPException(status_code=400, detail="Question or file required")

    file_context = ""
    search_hint_from_file: Optional[str] = None

    if upload_file:
        try:
            if upload_file.content_type.startswith("audio/"):
                tmp_path = os.path.join(TMP_AUDIO_DIR, f"{uuid.uuid4()}.wav")
                with open(tmp_path, "wb") as f:
                    f.write(await upload_file.read())

                transcribed = transcribe_audio_file(tmp_path, language_hint=language)
                file_context = f"\n[Audio Transcript]: {transcribed}\n"
                search_hint_from_file = transcribed
                question = question or transcribed
                os.remove(tmp_path)

            elif upload_file.content_type.startswith("text/"):
                content = await upload_file.read()
                decoded = content.decode("utf-8", errors="ignore")
                file_context = f"\n[Uploaded Text]: {decoded}\n"
                search_hint_from_file = decoded

            else:
                file_context = f"\n[Uploaded File: {upload_file.filename}]\n"

        except Exception as e:
            print("File handling error:", e)

    base_question = question or "Analyze the uploaded content"
    final_query = f"{base_question}\n{file_context}" if file_context else base_question

    rag_query = question or search_hint_from_file or base_question
    chunks = safe_search_patient_index(patient_id, query=rag_query, top_k=8)

    context = "\n\n---\n\n".join(
        [c["text"] for c in chunks if isinstance(c, dict) and c.get("text")]
    ) if chunks else ""

    if context:
        context = (
            "Patient records are ordered chronologically. "
            "Recent findings appear later in the text.\n\n"
            + context
        )

    if not context:
        lab_context = build_lab_context(db, patient_id)

        if lab_context:
            context = (
                "The following are structured lab results for this patient:\n\n"
                + lab_context
            )
        else:
            context = (
                "No medical reports are available yet for this patient."
            )

    # --- UPDATED PROMPT TO STOP MIRRORING ---
    system_prompt = f"""
        You are a medical assistant.

        CRITICAL LANGUAGE RULE (MANDATORY):
        - You MUST reply ONLY in the following language: {language}
        - If the user asks in Tamil, reply ONLY in Tamil.
        - DO NOT repeat the user's question. Start your response directly.
        - NEVER ask "What is your problem?" back to the user.

        MEDICAL RULES:
        - Use simple language if the user is a patient.
        - Use medical terminology if the user is a doctor.
        - Use ONLY the provided patient records.
        - If information is missing, clearly say you do not have enough data.
        - Do NOT prescribe medicines or dosages.
    """

    MAX_PROMPT_CHARS = 3500

    if len(context) > MAX_PROMPT_CHARS:
        context = context[-MAX_PROMPT_CHARS:]

    if len(final_query) > 1000:
        final_query = final_query[:1000]
    
    # FIXED: user_prompt defined BEFORE generate_text
    user_prompt = f"""
    Patient ID: {patient_id}

    Patient record snippets:
    {context if context else "[NO DATA AVAILABLE]"}

    User query:
    {final_query}

    FINAL CHECK:
    - Answer fully and directly in {language}.
    - DO NOT echo the question.
    """

    try:
        answer_text = generate_text(system_prompt, user_prompt, max_tokens=600)
    except Exception as e:
        print("LLM ERROR:", e)
        raise HTTPException(
            status_code=503,
            detail="LLM temporarily unavailable. Please try again."
        )

    audio_url = None
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = os.path.join(AUDIO_RESPONSE_DIR, audio_filename)

    if text_to_speech(answer_text, language, audio_path):
        audio_url = f"/storage/audio_responses/{audio_filename}"

    chat_entry = models.ChatHistory(
        patient_id=patient_id,
        asked_by_role=current_user.role,
        question=final_query,
        answer=answer_text,
    )
    db.add(chat_entry)
    db.commit()
    db.refresh(chat_entry)

    return schemas.ChatOut(
        id=chat_entry.id,
        patient_id=chat_entry.patient_id,
        asked_by_role=chat_entry.asked_by_role,
        question=chat_entry.question,
        answer=chat_entry.answer,
        created_at=chat_entry.created_at,
        audio_url=audio_url,
    )

@router.post("/{patient_id}/audio", response_model=schemas.ChatOut)
async def chat_with_patient_history_audio(
    patient_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if current_user.role == "patient" and current_user.patient_id != patient_id:
        raise HTTPException(status_code=403, detail="Not allowed")

    if current_user.role not in ["doctor", "patient"]:
        raise HTTPException(status_code=403, detail="Not allowed")

    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    tmp_path = os.path.join(TMP_AUDIO_DIR, f"{uuid.uuid4()}.wav")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        question_text = transcribe_audio_file(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not question_text:
        raise HTTPException(status_code=400, detail="Could not transcribe audio")

    chunks = safe_search_patient_index(patient_id, query=question_text, top_k=8)
    context = "\n\n---\n\n".join(
        [c["text"] for c in chunks if isinstance(c, dict) and c.get("text")]
    ) if chunks else ""

    if not context:
        context = (
            "No textual patient records are available yet. "
            "Only structured lab data exists."
        )

    system_prompt = (
        "You are a careful medical assistant. "
        "Use ONLY the provided patient records. "
        "Do NOT repeat the user's question."
    )

    MAX_CONTEXT_CHARS = 4000
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[-MAX_CONTEXT_CHARS:]

    # FIXED: user_prompt defined BEFORE generate_text
    user_prompt = f"""
    Patient record snippets:
    {context if context else "[NO DATA AVAILABLE]"}

    User question:
    {question_text}
    """

    try:
        answer_text = generate_text(system_prompt, user_prompt, max_tokens=600)
    except Exception as e:
        print("LLM ERROR:", e)
        raise HTTPException(
            status_code=503,
            detail="LLM temporarily unavailable. Please try again."
        )

    chat_entry = models.ChatHistory(
        patient_id=patient_id,
        asked_by_role=current_user.role,
        question=question_text,
        answer=answer_text,
    )
    db.add(chat_entry)
    db.commit()
    db.refresh(chat_entry)

    return schemas.ChatOut(
        id=chat_entry.id,
        patient_id=chat_entry.patient_id,
        asked_by_role=chat_entry.asked_by_role,
        question=chat_entry.question,
        answer=chat_entry.answer,
        created_at=chat_entry.created_at,
    )
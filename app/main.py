from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3

from .database import Base, engine, SessionLocal
from .models import User
from .routers.auth import get_password_hash
from .routers import (
    patients,
    reports,
    auth,
    summaries,
    chat,
    users,
    admin,
    analytics,
)

def migrate_db():
    """
    Adds content_hash column to reports table if it doesn't exist.
    Safe to run multiple times.
    """
    try:
        conn = sqlite3.connect("medical_rag.db")
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(reports)")
        columns = [column[1] for column in cursor.fetchall()]

        if "content_hash" not in columns:
            print("Migrating database: Adding content_hash to reports...")
            cursor.execute(
                "ALTER TABLE reports ADD COLUMN content_hash TEXT"
            )
            conn.commit()
    except Exception as e:
        print(f"Migration error: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

def create_initial_admin():
    """
    Default credentials:
      username: admin
      password: admin123
    """
    db = SessionLocal()
    try:
        existing_admin = db.query(User).filter(User.role == "admin").first()
        if existing_admin:
            return

        admin_user = User(
            username="admin",
            password_hash=get_password_hash("admin123"),
            role="admin",
            patient_id=None,
        )
        db.add(admin_user)
        db.commit()
        print("Created default admin user: username='admin', password='admin123'")
    finally:
        db.close()

migrate_db()

Base.metadata.create_all(bind=engine)

create_initial_admin()

app = FastAPI(
    title="Medical RAG Backend",
    version="0.2.0",
)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(reports.router)
app.include_router(summaries.router)
app.include_router(chat.router)
app.include_router(users.router)
app.include_router(admin.router)
app.include_router(analytics.router)

@app.get("/")
def root():
    return {"message": "Medical RAG backend is running"}
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..deps import get_db
from .auth import get_current_user

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/", response_model=List[schemas.UserOut])
def list_users(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    # Simple guard: only doctors (or some 'admin' role) can see all users
    if current_user.role not in ("doctor", "admin"):
        raise HTTPException(status_code=403, detail="Not allowed to list users")

    users = db.query(models.User).all()
    return users

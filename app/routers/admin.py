from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import models, schemas
from ..deps import get_db
from .auth import get_current_user, get_password_hash  

router = APIRouter(prefix="/admin", tags=["admin"])


def get_current_admin(
    current_user: models.User = Depends(get_current_user),
) -> models.User:
    """
    Dependency: only allow users with role='admin'.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admins only",
        )
    return current_user


@router.get("/users", response_model=List[schemas.UserOut])
def admin_list_users(
    db: Session = Depends(get_db),
    admin: models.User = Depends(get_current_admin),
):
    users = db.query(models.User).all()
    return users


@router.post("/users", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def admin_create_user(
    user_in: schemas.UserCreate,
    db: Session = Depends(get_db),
    admin: models.User = Depends(get_current_admin),
):

    # Check username uniqueness
    existing = db.query(models.User).filter(models.User.username == user_in.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken",
        )

    hashed_pw = get_password_hash(user_in.password)

    db_user = models.User(
        username=user_in.username,
        password_hash=hashed_pw,
        role=user_in.role,
        patient_id=user_in.patient_id,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

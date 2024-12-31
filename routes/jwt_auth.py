from fastapi import APIRouter, Body
from models.auth import (
    UserSignUpSchema,
    UserLoginSchema
)
from utils.jwt_handler import signJWT
from utils.check_user_existance import check_user

virtual_db_users = []
router = APIRouter()

# @router.post("/signup")
def create_user(user: UserSignUpSchema = Body(...)):
    virtual_db_users.append(user) # replace with db call, making sure to hash the password first
    return signJWT(user.fullname)

@router.post("/login")
def user_login(user: UserLoginSchema = Body(...)):
    if check_user(user, virtual_db_users):
        return signJWT(user.fullname)
    return {
        "error": "Wrong login details!"
    }
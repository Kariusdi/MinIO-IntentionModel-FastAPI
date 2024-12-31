from models.auth import (
    UserLoginSchema
)

def check_user(data: UserLoginSchema, users):
    for user in users:
        if user.fullname == data.fullname and user.password == data.password:
            return True
    return False
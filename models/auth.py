from pydantic import BaseModel, Field, EmailStr

class UserSignUpSchema(BaseModel):
    fullname: str = Field(...)
    # email: EmailStr = Field(...)
    password: str = Field(...)

    class Config:
        json_schema_extra = {
            "example": {
                "fullname": "Chonakan",
                # "email": "chonakan@example.com",
                "password": "1234"
            }
        }

class UserLoginSchema(BaseModel):
    fullname: str = Field(...)
    password: str = Field(...)

    class Config:
        json_schema_extra = {
            "example": {
                "fullname": "Chonakan",
                "password": "1234"
            }
        }
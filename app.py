from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.db import router as DbRouter
from routes.jwt_auth import router as JwtAuth

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(DbRouter, tags=["MinIO Management (CRUD)"], prefix="/minio")
app.include_router(JwtAuth, tags=["Authentication JWT"], prefix="/auth")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Hello world!, This is MinIO REST API server"}
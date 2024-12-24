from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from db.crud import list_files, list_delta_files, delete_latest_version, get_file
import zipfile
import io
from db.minio_connection import MinioClient
import asyncio
from minio.error import S3Error
from typing import Optional

router = APIRouter()
bucket_name = "cdti-policies-md"

minio = MinioClient(host="localhost:9000", 
                    access_key="qOq0EgEOlA86lXSi9vcg", 
                    secret_key="Oex54aRLb6HhIeEI7HJrY8SW1PrLqi1H06HsSJ2q",
                    secure_bool=False)
client = minio.get_client()

@router.get("/list")
async def files_lister():
    """
    API endpoint to list all files in the bucket.
    """
    files = await list_files(bucket_name)
    return {"files": files, "total": len(files)}

@router.get("/delta/{folder}/{file_name}")
async def delta_files_lister(folder: str, file_name: str):
    """
    API endpoint to list a files with a specific path in the bucket.
    """
    files = await list_delta_files(bucket_name, folder, file_name)
    return {"files": files, "total": len(files)}

@router.get("/list/{folder}/{file_name}")
async def file_getter(folder: str, file_name: str, version_id: str = None):
    """
    API endpoint to retrieve specific file versions from a folder as a ZIP file.
    """
    baseline_key = f"{folder}/{file_name}/baseline.md"
    delta_key = f"{folder}/{file_name}/delta.bsdiff"
    
    baseline_res = await get_file(bucket_name, baseline_key, version_id)
    delta_res = await get_file(bucket_name, delta_key, version_id)

    baseline_content = baseline_res.read()
    delta_content = delta_res.read()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("baseline.md", baseline_content)
        zip_file.writestr("delta.bsdiff", delta_content)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={file_name}.zip"}
    )

@router.delete("/delta/{folder}/{file_name}")
async def latest_delta_file_deleter(folder: str, file_name: str):
    """
    API endpoint to delete specific file versions from a folder in a bucket.
    """
    object_delta = f"{folder}/{file_name}/delta.bsdiff"
    await delete_latest_version(bucket_name, object_delta)

    return {"status": 200, "detial": "Delete Latest Delta File Successfully."}

@router.post("/upload-file/")
async def upload_file(
    bucket_name: str = Form(...),  # The bucket to upload to
    file: UploadFile = Form(...),  # The file to upload
    object_name: Optional[str] = Form(None),  # Optional custom object name
):
    """
    API endpoint to upload a file to MinIO.
    - `bucket_name`: Name of the bucket to upload the file.
    - `file`: File to be uploaded (sent as multipart form data).
    - `object_name`: Optional name for the file in the bucket. Defaults to the uploaded file's name.
    """
    try:
        # If no object name is provided, use the uploaded file's name
        object_name = object_name or file.filename

        # Upload the file to the specified bucket
        response = await asyncio.to_thread(
            client.put_object,
            bucket_name,
            object_name,
            file.file,  # The file's content
            length=-1,  # Let MinIO calculate the length automatically
            content_type=file.content_type,  # Content type of the uploaded file
        )
        return {"message": "File uploaded successfully", "object_name": object_name}
    except S3Error as e:
        print("Error uploading file:", e)
        raise HTTPException(status_code=500, detail="File upload failed.")
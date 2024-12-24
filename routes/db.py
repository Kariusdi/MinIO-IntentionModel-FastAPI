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
from io import BytesIO
from dotenv import load_dotenv

ACCESS_KEY = load_dotenv("ACCESS_KEY")
SECRET_KEY = load_dotenv("SECRET_KEY")
HOST = str(load_dotenv("DEV_HOST"))

router = APIRouter()

minio = MinioClient(host=HOST, 
                    access_key=ACCESS_KEY, 
                    secret_key=SECRET_KEY,
                    secure_bool=False)
client = minio.get_client()

@router.get("/list")
async def files_lister(bucket_name: str):
    """
    API endpoint to list all files in the bucket.
    """
    files = await list_files(bucket_name)
    return {"files": files, "total": len(files)}

@router.get("/delta")
async def delta_files_lister(bucket_name: str, folder: str, file_name: str):
    """
    API endpoint to list a files with a specific path in the bucket.
    """
    files = await list_delta_files(bucket_name, folder, file_name)
    return {"files": files, "total": len(files)}

@router.get("/get")
async def file_getter(bucket_name: str, folder: str, file_name: str, version_id: str = None):
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

@router.delete("/delta")
async def latest_delta_file_deleter(bucket_name: str, folder: str, file_name: str):
    """
    API endpoint to delete specific file versions from a folder in a bucket.
    """
    object_delta = f"{folder}/{file_name}/delta.bsdiff"
    await delete_latest_version(bucket_name, object_delta)

    return {"status": 200, "detial": "Delete Latest Delta File Successfully."}

@router.post("/upload-zip")
async def upload_files(file: UploadFile = File(...)):
    file_content = await file.read()
    success_files = []
    try:
        with zipfile.ZipFile(BytesIO(file_content), 'r') as zip_ref:
            for zip_file_name in zip_ref.namelist():
                if zip_file_name.endswith('/') or zip_file_name.startswith('__MACOSX/'):
                    continue
                try:
                    with zip_ref.open(zip_file_name) as document:
                        document_content = document.read()
                        object_name = "syllabus/" + zip_file_name
                        client.put_object(
                            bucket_name="cdti-policies",
                            object_name=object_name,
                            data=BytesIO(document_content),  # Wrap in BytesIO for streaming
                            length=len(document_content),
                            content_type="application/octet-stream"  # Default content type
                        )
                        success_files.append(zip_file_name)
                except S3Error as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error uploading file: {str(e)}"
                    )
                except Exception as e:
                    print(f"Error uploading {zip_file_name}: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error uploading {zip_file_name}: {str(e)}.")
        return {"status": 200, "message": f"ZIP file processed and {len(success_files)} files uploaded successfully!"}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="A file is not a valid ZIP file. Use ZIP file format only with this endpoint.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ZIP file: {str(e)}")
    

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    API endpoint to upload a file to the MinIO bucket.
    """
    try:
        object_name = "syllabus/" + file.filename
        client.put_object(
            bucket_name="cdti-policies",
            object_name=object_name,
            data=file.file,
            length=file.size,
            content_type=file.content_type
        )

        return {
            "status": 200, 
            "message": f"File uploaded to {object_name} successfully!",
        }
    except S3Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
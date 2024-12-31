from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import zipfile
import io
from io import BytesIO
from db.crud import (
    list_files, 
    list_delta_files, 
    delete_latest_version, 
    get_file, 
    upload_file
)
from models.response import (
    SuccessListResponse,
    SuccessResponse
)
from utils.jwt_bearer import JWTBearer

router = APIRouter()

@router.get("/list", dependencies=[Depends(JWTBearer())])
async def files_lister(bucket_name: str):
    """
    ## <mark>API endpoint to list all files in the bucket (not a specific folder).
    
    #### **Request Parameters** 
    | Parameter   | Type   | Description                | Required |
    |-------------|--------|----------------------------|----------|
    | `bucket`    | string | Name of the bucket         | Yes      |

    #### **Response Example**
    ```json
    {
        "status": 200,
        "data": {
            "folder": "None",
            "file_name": "None",
            "delta_files": [
                {
                     "bucket_name": "cdti-policies",
                    "object_name": "guide/student_guide.pdf",
                    "size": 3128851,
                    "version_id": "0ddd9a14-2b43-4e70-a3e0-6e73962fcb32",
                    "is_latest": "true"
                }
            ]
        },
        "total": 1
    }
    """
    try:
        files = await list_files(bucket_name)
        return SuccessListResponse(data=files)
    except HTTPException as e:
        raise e
        
@router.get("/delta", dependencies=[Depends(JWTBearer())])
async def delta_files_lister(folder: str, file_name: str):
    """
    ## <mark>API endpoint to list delta files with a specific folder and filename in the `cdti-policies-md collection`. 
    
    #### **Request Parameters** 
    | Parameter   | Type   | Description                                      | Required |
    |-------------|--------|--------------------------------------------------|----------|
    | `folder`    | string | Name of the folder within the bucket.            | Yes      |
    | `file_name` | string | Name of the specific file to search for deltas.  | Yes      |

    #### **Response Example**
    ```json
    {
        "status": 200,
        "data": {
            "folder": "syllabus",
            "file_name": "course.pdf",
            "delta_files": [
                {
                    "bucket_name": "cdti-policies-md",
                    "object_name": "syllabus/course.pdf/delta.bsdiff",
                    "size": 142,
                    "version_id": "b0e6df27-bc86-41be-a35b-f1d2f50864f1",
                    "is_latest": "true"
                }
            ]
        },
        "total": 1
    }
    """
    try:
        files = await list_delta_files(folder, file_name)
        if not files:
            return SuccessListResponse(data="There's no delta files in this file_name", folder=folder, file_name=file_name)
        return SuccessListResponse(data=files, folder=folder, file_name=file_name)
    except HTTPException as e:
        raise e

@router.get("/raw", dependencies=[Depends(JWTBearer())])
async def raw_file_getter(folder: str, file_name: str, version: str = "0"):
    """
    ## <mark>API endpoint to retrieve specific raw file (original document) from a folder and return it as a Zip file.
    
    #### **Request Parameters** 
    | Parameter   | Type   | Description                                      | Required |
    |-------------|--------|--------------------------------------------------|----------|
    | `folder`    | string | Name of the folder within the bucket.            | Yes      |
    | `file_name` | string | Name of the specific file to search for deltas.  | Yes      |
    | `version`   | string | Version of the file.                             | No       |

    #### **Response Example**
    Zip file
    """
    bucket_name = "cdti-policies"
    object_key = f"{folder}/{file_name}"
    
    try:
        file_content = await get_file(bucket_name, object_key, version)
    except HTTPException as e:
        raise e
    
    return StreamingResponse(
        file_content,
        media_type="application/octet-stream",  # Use the appropriate MIME type for your file
        headers={"Content-Disposition": f"attachment; filename={file_name}"}
    )

@router.get("/transfromed", dependencies=[Depends(JWTBearer())])
async def transformed_file_getter(folder: str, file_name: str, version: str = "0"):
    """
    ## <mark>API endpoint to retrieve specific transformed file (cleaned document) from a folder and return it as a Zip file.
    
    #### **Request Parameters** 
    | Parameter   | Type   | Description                                      | Required |
    |-------------|--------|--------------------------------------------------|----------|
    | `folder`    | string | Name of the folder within the bucket.            | Yes      |
    | `file_name` | string | Name of the specific file to search for deltas.  | Yes      |
    | `version`   | string | Version of the file.                             | No       |

    #### **Response Example**
    Zip file
    """
    bucket_name = "cdti-policies-md"
    baseline_key = f"{folder}/{file_name}/baseline.md"
    delta_key = f"{folder}/{file_name}/delta.bsdiff"
    
    try:
        baseline_res = await get_file(bucket_name, baseline_key, version=None)
        delta_res = await get_file(bucket_name, delta_key, version)
    except HTTPException as e:
        raise e
        
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

@router.delete("/delta", dependencies=[Depends(JWTBearer())])
async def latest_delta_file_deleter(folder: str, file_name: str):
    """
    ## <mark>API endpoint to delete a specific file (original document) from a folder in the `cdti-policies-md collection`.
    
    #### **Request Parameters** 
    | Parameter   | Type   | Description                                      | Required |
    |-------------|--------|--------------------------------------------------|----------|
    | `folder`    | string | Name of the folder within the bucket.            | Yes      |
    | `file_name` | string | Name of the specific file to search for deltas.  | Yes      |

    #### **Response Example**
    ```json
    {
        "status": 200,
        "detail": "Delete Latest Version of syllabus/course.pdf/delta.bsdiff Successfully.",
        "total": 0
    }
    """
    try:
        bucket_name = "cdti-policies-md"
        object_delta = f"{folder}/{file_name}/delta.bsdiff"
        res = await delete_latest_version(bucket_name, object_delta)
        if res:
            return SuccessResponse(details=f"Delete {object_delta} File Successfully.")
    except HTTPException as e:
        raise e

@router.delete("/raw", dependencies=[Depends(JWTBearer())])
async def latest_delta_file_deleter(folder: str, file_name: str):
    """
    ## <mark>API endpoint to delete a specific file (original document) from a folder in the `cdti-policies collection`.
    
    #### **Request Parameters** 
    | Parameter   | Type   | Description                                      | Required |
    |-------------|--------|--------------------------------------------------|----------|
    | `folder`    | string | Name of the folder within the bucket.            | Yes      |
    | `file_name` | string | Name of the specific file to search for deltas.  | Yes      |

    #### **Response Example**
    ```json
    {
        "status": 200,
        "detail": "Delete Latest Version of syllabus/course.pdf Successfully.",
        "total": 0
    }
    """
    try:
        bucket_name = "cdti-policies"
        object_raw = f"{folder}/{file_name}"
        res = await delete_latest_version(bucket_name, object_raw)
        if res:
            return SuccessResponse(data=f"Delete {object_raw} File Successfully.")
    except HTTPException as e:
        raise e
    
@router.post("/upload", dependencies=[Depends(JWTBearer())])
async def file_uploader(bucket_name: str, folder: str, file: UploadFile = File(...)):
    """
    ## <mark>API endpoint to upload a file to a specific bucket and folder.
    
    #### **Request Parameters** 
    | Parameter     | Type   | Description                                      | Required |
    |---------------|--------|--------------------------------------------------|----------|
    | `bucket_name` | string | Name of the bucket.                              | Yes      |
    | `folder`      | string | Name of the folder within the bucket.            | Yes      |
    | `file`        | File   | File that you'd like to upload                   | Yes      |

    #### **Response Example**
    ```json
    {
        "status": 200,
        "details": "ZIP file processed and 2 files uploaded successfully! ['course.pdf', 'course2.pdf']"
    }
    """
    if file.filename.endswith(('.pdf', '.docx')):
        try:
            object_name = f"{folder}/{file.filename}"
            res = await upload_file(
                bucket_name=bucket_name,
                object_name=object_name,
                data=file.file,
                length=file.size,
                content_type=file.content_type
            )
            if res:
                return SuccessResponse(details=f"File uploaded to {object_name} successfully!")
        except HTTPException as e:
            raise e
    else:
        raise HTTPException(
            status_code=400,
            detail=f".pdf or .docx types are only allowed. The '{file.filename}' is not allowed here!"
        )

@router.post("/upload-files", dependencies=[Depends(JWTBearer())])
async def files_uploader(bucket_name: str, folder: str, file: UploadFile = File(...)):
    """
    ## <mark>API endpoint to upload files (Zip file) to a specific bucket and folder.
    
    #### **Request Parameters** 
    | Parameter     | Type   | Description                                      | Required |
    |---------------|--------|--------------------------------------------------|----------|
    | `bucket_name` | string | Name of the bucket.                              | Yes      |
    | `folder`      | string | Name of the folder within the bucket.            | Yes      |
    | `file`        | File   | File that you'd like to upload                   | Yes      |

    #### **Response Example**
    ```json
    {
        "status": 200,
        "detail": "ZIP file processed and 2 files uploaded successfully!",
    }
    """
    file_content = await file.read()
    success_files = []
    try:
        with zipfile.ZipFile(BytesIO(file_content), 'r') as zip_ref:
            for zip_file_name in zip_ref.namelist():
                if zip_file_name.endswith('/') or zip_file_name.startswith('__MACOSX/'):
                    continue
                with zip_ref.open(zip_file_name) as document:
                    try:
                        document_content = document.read()
                        object_name = f"{folder}/{zip_file_name}"
                        _ = await upload_file(
                            bucket_name=bucket_name,
                            object_name=object_name,
                            data=BytesIO(document_content),
                            length=len(document_content),
                            content_type="application/octet-stream"
                        )
                        success_files.append(zip_file_name)
                    except HTTPException as e:
                        raise e
        return SuccessResponse(details=f"ZIP file processed and {len(success_files)} files uploaded successfully! {success_files}")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="A file is not a valid ZIP file. Use ZIP file format only with this endpoint.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ZIP file: {str(e)}")
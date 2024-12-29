from fastapi import HTTPException
from minio.error import S3Error
from db.minio_connection import MinioClient
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
HOST = os.getenv('DEV_HOST')

minio = MinioClient(host=HOST, 
                    access_key=ACCESS_KEY, 
                    secret_key=SECRET_KEY,
                    secure_bool=False)
client = minio.get_client()

def listfiles_helper(obj):
    return {
        "bucket_name": obj.bucket_name, 
        "object_name": obj.object_name, 
        "size": obj.size, 
        "version_id": obj.version_id, 
        "is_latest": obj.is_latest
        }

async def list_files(bucket_name):
    try:
        objects = await asyncio.to_thread(client.list_objects, bucket_name, recursive=True, include_version=True)
        return [listfiles_helper(obj) for obj in objects]
    except S3Error as exc:
        print("Error listing objects:", exc)
        raise HTTPException(status_code=500, detail="Could not list files in the bucket. It's empty.")

async def list_delta_files(folder, file_path):
    try:
        bucket_name = "cdti-policies-md"
        objects = await asyncio.to_thread(client.list_objects, bucket_name, prefix=f"{folder}/{file_path}/delta.bsdiff", include_version=True)
        if objects:
            return [listfiles_helper(obj) for obj in objects]
        else:
            return False
    except S3Error as exc:
        print("Error listing objects:", exc)
        raise HTTPException(status_code=500, detail="Could not list files in the bucket. It's empty.")

async def get_file(bucket_name, object_name, version=0):
    try:
        versions = await list_file_versions(bucket_name, object_name)
        if not versions:
            raise HTTPException(status_code=404, detail="No versions found for the object.")

        if version != None:
            latest_version = versions[int(version)]
            latest_version_id = latest_version.version_id
        else:
            latest_version_id = None
        
        response = await asyncio.to_thread(
            client.get_object, bucket_name, object_name, version_id=latest_version_id
        )
        return response
    except IndexError:
        raise HTTPException(status_code=500, detail=f"This version of the {object_name} doesn't exist.")
    except S3Error as exc:
        raise HTTPException(status_code=404, detail=f"File or version not found. {exc}")

async def delete_file(bucket_name, object_name, version_id=None):
    try:
        response = await asyncio.to_thread(
            client.remove_object, bucket_name, object_name, version_id=version_id
        )
        return response
    except S3Error as exc:
        print("Error removing object:", exc)
        raise HTTPException(status_code=404, detail="File or version not found.")
    
async def list_file_versions(bucket_name, object_name):
    try:
        response = await asyncio.to_thread(client._list_objects, bucket_name, prefix=object_name, include_version=True)
        return [version for version in response if version.object_name == object_name]
    except S3Error as exc:
        print("Error listing object versions:", exc)
        return []

async def delete_latest_version(bucket_name, object_name):
    try:
        versions = await list_file_versions(bucket_name, object_name)
        if not versions:
            raise HTTPException(status_code=404, detail="No versions found for the object.")

        latest_version = versions[0]
        latest_version_id = latest_version.version_id
        
        await delete_file(bucket_name, object_name, version_id=latest_version_id)
        print(f"Deleted latest version: {latest_version_id}")

        if len(versions) > 1 and versions[1].is_delete_marker:
            delete_marker_version_id = versions[1].version_id
            await delete_file(bucket_name, object_name, version_id=delete_marker_version_id)
            print(f"Deleted delete marker: {delete_marker_version_id}")
        
        return True
    except HTTPException as e:
        print("Failed to delete latest version:", e.detail)

async def upload_file(bucket_name, object_name, data, length, content_type):
    try:
        await asyncio.to_thread(
            client.put_object, bucket_name, object_name, data, length, content_type
        )
        return True
    except S3Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file into the bucket: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

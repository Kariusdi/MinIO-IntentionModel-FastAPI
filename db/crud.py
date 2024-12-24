from fastapi import HTTPException
from minio.error import S3Error
from db.minio_connection import MinioClient
from minio.deleteobjects import DeleteObject
import asyncio

minio = MinioClient(host="localhost:9000", 
                    access_key="qOq0EgEOlA86lXSi9vcg", 
                    secret_key="Oex54aRLb6HhIeEI7HJrY8SW1PrLqi1H06HsSJ2q",
                    secure_bool=False)
client = minio.get_client()

async def list_files(bucket_name):
    try:
        objects = await asyncio.to_thread(client.list_objects, bucket_name, recursive=True, include_version=True)
        return [{"bucket_name": obj.bucket_name, 
                 "object_name": obj.object_name, 
                 "size": obj.size, 
                 "version_id": obj.version_id, 
                 "is_latest": obj.is_latest} for obj in objects]
    except S3Error as exc:
        print("Error listing objects:", exc)
        raise HTTPException(status_code=500, detail="Could not list files in the bucket. It's empty.")

async def list_delta_files(bucket_name, folder, file_path):
    try:
        objects = await asyncio.to_thread(client.list_objects, bucket_name, prefix=f"{folder}/{file_path}/delta.bsdiff", include_version=True)
        return [{"bucket_name": obj.bucket_name, 
                 "object_name": obj.object_name, 
                 "size": obj.size, 
                 "version_id": obj.version_id, 
                 "is_latest": obj.is_latest} for obj in objects]
    except S3Error as exc:
        print("Error listing objects:", exc)
        raise HTTPException(status_code=500, detail="Could not list files in the bucket. It's empty.")

async def get_file(bucket_name, object_name, version_id=None):
    try:
        response = await asyncio.to_thread(
            client.get_object, bucket_name, object_name, version_id=version_id
        )
        return response
    except S3Error as exc:
        print("Error retrieving object:", exc)
        raise HTTPException(status_code=404, detail="File or version not found.")

async def delete_file(bucket_name, object_name, version_id=None):
    try:
        response = await asyncio.to_thread(
            client.remove_object, bucket_name, object_name, version_id=version_id
        )
        return response
    except S3Error as exc:
        print("Error removing object:", exc)
        raise HTTPException(status_code=404, detail="File or version not found.")
    
async def list_object_versions(bucket_name, object_name):
    try:
        response = await asyncio.to_thread(client._list_objects, bucket_name, prefix=object_name, include_version=True)
        return [version for version in response if version.object_name == object_name]
    except S3Error as exc:
        print("Error listing object versions:", exc)
        return []

async def delete_latest_version(bucket_name, object_name):
    try:
        # Step 1: List all versions of the object
        versions = await list_object_versions(bucket_name, object_name)
        if not versions:
            raise HTTPException(status_code=404, detail="No versions found for the object.")

        # Step 2: Find the latest version
        latest_version = versions[0]  # The first item should be the latest version
        latest_version_id = latest_version.version_id

        # Step 3: Delete the latest version
        await delete_file(bucket_name, object_name, version_id=latest_version_id)
        print(f"Deleted latest version: {latest_version_id}")

        # Step 4: Handle the delete marker if needed
        # Check if the next version exists and is a delete marker
        if len(versions) > 1 and versions[1].is_delete_marker:
            delete_marker_version_id = versions[1].version_id
            await delete_file(bucket_name, object_name, version_id=delete_marker_version_id)
            print(f"Deleted delete marker: {delete_marker_version_id}")

    except HTTPException as e:
        print("Failed to delete latest version:", e.detail)
        

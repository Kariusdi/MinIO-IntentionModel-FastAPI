def SuccessResponse(status_code: int = 200 , folder: str = "None", file_name: str = "None", data = []):
    total = 0 if type(data) != "list" else len(data)
    return {"status": status_code,
            "data": {
                "folder": folder, 
                "file_name": file_name,
                "data": data, 
            },
            "total": total
            }

def ExceptionResponse(status_code: int = 500 ,data = 0):
    return {"status": status_code, 
            "details": data}
def SuccessResponse(status_code: int = 200 ,data: int = 0):
    return {"status": status_code, 
            "details": data, 
            "total": len(data)}
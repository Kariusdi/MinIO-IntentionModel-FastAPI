def SuccessResponse(status_code: int = 200 ,details = 0):
    total = 0 if type(details) != "list" else len(details)
    return {"status": status_code, 
            "details": details, 
            "total": total}

def ExceptionResponse(status_code: int = 500 ,data = 0):
    return {"status": status_code, 
            "details": data}
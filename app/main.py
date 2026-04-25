from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os

from app.detector import detect_patterns, model_meta

app = FastAPI(
    title="TradePatterns API",
    description="Classical candlestick pattern detection using Random Forest",
    version="1.0.0"
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class DetectRequest(BaseModel):
    ticker:     str = "AAPL"
    start_date: str = "2024-01-01"
    end_date:   str = "2024-12-31"

@app.get("/", include_in_schema=False)
def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "TradePatterns API is running"}

@app.get("/model-info")
def model_info():
    return JSONResponse(content=model_meta)

@app.post("/detect")
def detect(request: DetectRequest):
    try:
        result = detect_patterns(
            ticker=request.ticker.upper(),
            start=request.start_date,
            end=request.end_date
        )
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
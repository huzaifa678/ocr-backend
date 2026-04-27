from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from src.logger import setup_logger
from src.model import ExtractResponse
from src.extract import extract_text_from_image


app = FastAPI()
logger = setup_logger()

@app.post("/extract")
async def extract_text(file: UploadFile = File(...), option: Literal["easy_ocr", "paddle"] = "paddle") -> ExtractResponse:

    if file.content_type not in {"image/jpeg", "image/jpg"}:
        logger.warning("Rejected file due to invalid MIME type.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG/JPEG images are allowed."
        )

    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        logger.warning("Rejected file due to invalid extension.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have .jpg or .jpeg extension."
        )

    image_bytes = await file.read()
    
    try:
        logger.info(f"Received file '{file.filename}' for text extraction.")
        results = extract_text_from_image(image_bytes, option)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the image."
        )
    
    return ExtractResponse(
        results=results.texts
    )

import io
from typing import Literal
from PIL import Image
import numpy as np
from easyocr import easyocr
from paddleocr import PaddleOCR
from src.logger import setup_logger
from src.model import ExtractResult

reader = easyocr.Reader(['en'], gpu=False)
logger = setup_logger()

def extract_text_from_image(image_bytes: bytes, option: Literal["ocr", "paddle"]) -> ExtractResult:
    
    logger.info("Starting text extraction from image.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info("Image loaded successfully for processing.")
    except Exception as exc:
        logger.warning("Invalid image received for inference.")
        raise ValueError("Invalid image file.") from exc
    
    image_np = np.array(image)

    if option == "paddle":
        logger.info("Using PaddleOCR for text extraction.")
        paddleOcr = PaddleOCR(use_angle_cls=True, lang='en')
        logger.info("PaddleOCR model loaded successfully.")
        results = paddleOcr.ocr(image_np)
        print(results[0])
        logger.info(f"Extracted {len(results[0]['rec_texts'])} text lines using PaddleOCR.")
        results = results[0]['rec_texts']
    else:
        logger.info("Using EasyOCR for text extraction.")
        results = reader.readtext(image_np)
        results = [text for _, text, _ in results]
        logger.info(f"Extracted {len(results)} text lines using EasyOCR.")

    return ExtractResult(texts=results)

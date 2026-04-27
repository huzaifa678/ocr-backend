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

def extract_text_from_image(image_bytes: bytes, option: Literal["easy_ocr", "paddle"]) -> ExtractResult:
    """
    This function takes the image bytes and the OCR option, converts back to the image, and runs OCR to extract text from the image.

    Parameters
    ----------
    Param: bytes
        a single image stream of bytes
    Param: option
        a string that specifies which OCR method to use ("easy_ocr" or "paddle)

    Returns
    -------
    List[str]
        Returns the extracted text lines

    Raise
    -----
    ValueError
        If the image file is invaid
    """
    
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
        logger.info(f"Extracted {len(results[0]['rec_texts'])} text lines using PaddleOCR.")
        results = results[0]['rec_texts']
    else:
        logger.info("Using EasyOCR for text extraction.")
        results = reader.readtext(image_np)
        results = [text for _, text, _ in results]
        logger.info(f"Extracted {len(results)} text lines using EasyOCR.")

    return ExtractResult(texts=results)

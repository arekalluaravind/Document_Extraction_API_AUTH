
import logging
import uuid
import os
import secrets
from pathlib import Path
from typing import Optional
from io import BytesIO

from dotenv import load_dotenv

# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Header,
    Depends,
)

from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageSequence

# Agentic workflow imports
from graph_workflow_updated import create_document_router_graph
from main_updated import save_result_to_excel

# Non-Agentic AWS Textract imports
from AwsTextract_Hybrid import (
    process_file,
    save_invoice_to_excel,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_KEY = os.getenv("API_KEY")

INPUT_DIR = BASE_DIR / "input_documents"
OUTPUT_DIR = BASE_DIR / "output_results"

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
}

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Document Extractor API",
    description="Agentic and Non-Agentic Invoice Extraction API",
    version="4.0.0",
)


# --------------------------------------------------
# API KEY AUTHENTICATION
# --------------------------------------------------

async def verify_api_key(
    x_api_key: Optional[str] = Header(default=None),
):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key is required.",
        )

    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key.",
        )

    return True


# --------------------------------------------------
# IMAGE TO PDF CONVERSION
# Used ONLY for Agentic mode
# --------------------------------------------------

def convert_image_to_pdf(
    image_path: Path,
    pdf_path: Path,
):
    """
    Convert PNG/JPG/JPEG images into a PDF.

    Supports multi-frame images where Pillow can read them.
    Non-Agentic mode does not use this function.
    """

    frames = []

    try:
        with Image.open(image_path) as image:

            for frame in ImageSequence.Iterator(image):

                converted_frame = frame.convert("RGB")
                frames.append(converted_frame.copy())

        if not frames:
            raise ValueError("No valid image frames found.")

        frames[0].save(
            pdf_path,
            format="PDF",
            save_all=True,
            append_images=frames[1:],
        )

        logger.info(
            "Image successfully converted to PDF: %s",
            pdf_path,
        )

    except Exception as exc:
        logger.exception("Image to PDF conversion failed.")

        raise ValueError(
            f"Unable to convert image to PDF: {str(exc)}"
        ) from exc

    finally:
        for frame in frames:
            frame.close()


# --------------------------------------------------
# AGENTIC DOCUMENT PROCESSING
# --------------------------------------------------

def process_agentic_document(
    pdf_path: Path,
    job_id: str,
    user_query: Optional[str] = None,
):

    logger.info("Starting AGENTIC document extraction")
    logger.info("Agentic input file: %s", pdf_path)

    graph = create_document_router_graph()

    initial_state = {
        "pdf_path": str(pdf_path),
        "user_query": user_query,
        "extracted_text": "",
        "extraction_method": "",
        "uploaded_file": None,
        "document_type": None,
        "extracted_data": None,
    }

    result = graph.invoke(initial_state)

    extracted_data = result.get("extracted_data")

    if extracted_data is None:
        raise ValueError(
            "Agentic workflow did not return extracted data."
        )

    output_folder = OUTPUT_DIR / job_id
    output_folder.mkdir(parents=True, exist_ok=True)

    save_result_to_excel(
        extracted_data=extracted_data,
        output_dir=str(output_folder),
        base_filename=pdf_path.stem,
    )

    excel_path = (
        output_folder /
        f"{pdf_path.stem}_extracted.xlsx"
    )

    if not excel_path.is_file():
        raise FileNotFoundError(
            f"Agentic Excel file was not created: {excel_path}"
        )

    if hasattr(extracted_data, "model_dump"):
        extracted_data_response = extracted_data.model_dump()
    else:
        extracted_data_response = extracted_data

    return {
        "document_type": result.get("document_type"),
        "extraction_method": result.get("extraction_method"),
        "extracted_data": extracted_data_response,
        "excel_file": excel_path.name,
    }


# --------------------------------------------------
# NON-AGENTIC DOCUMENT PROCESSING
# Supports PDF, PNG, JPG, JPEG directly
# --------------------------------------------------

def process_nonagentic_document(
    input_path: Path,
    job_id: str,
):

    logger.info("Starting NON-AGENTIC AWS Textract extraction")
    logger.info("Input file: %s", input_path)
    logger.info("Input extension: %s", input_path.suffix.lower())

    # Log file signature for debugging
    with open(input_path, "rb") as file:
        file_header = file.read(10)

    logger.info("Input file header: %s", file_header)

    # IMPORTANT:
    # Pass the original file path to process_file.
    # AwsTextract_Hybrid.py handles PDFs and images itself.

    textract_result = process_file(str(input_path))

    if not textract_result:
        raise ValueError(
            "AWS Textract did not return any extraction result."
        )

    output_folder = OUTPUT_DIR / job_id
    output_folder.mkdir(parents=True, exist_ok=True)

    save_invoice_to_excel(
        filename=input_path.name,
        result=textract_result,
        output_folder_path=str(output_folder),
    )

    excel_path = (
        output_folder /
        input_path.stem /
        "extracted_invoice_data.xlsx"
    )

    if not excel_path.is_file():
        raise FileNotFoundError(
            f"Non-Agentic Excel file was not created: {excel_path}"
        )

    return {
        "document_type": "invoice",
        "extraction_method": "AWS Textract - Non-Agentic",
        "extracted_data": textract_result,
        "excel_file": excel_path.name,
    }


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
async def root():

    return {
        "message": "Document Extractor API is running.",
        "version": "4.0.0",
        "available_modes": [
            "agentic",
            "nonagentic",
        ],
        "supported_files": [
            "PDF",
            "PNG",
            "JPG",
            "JPEG",
        ],
        "authentication": "API Key required for extraction and download",
        "docs": "/docs",
        "health": "/health",
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "message": "Document Extractor API is running.",
    }


# --------------------------------------------------
# DOCUMENT EXTRACTION ENDPOINT
# --------------------------------------------------

@app.post("/api/v1/documents/extract")
async def extract_document(

    file: UploadFile = File(...),

    extraction_mode: str = Form(...),

    user_query: Optional[str] = Form(default=None),

    authenticated: bool = Depends(verify_api_key),

):

    original_filename = Path(
        file.filename or "document.pdf"
    ).name

    file_ext = Path(original_filename).suffix.lower()

    # Validate file extension
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Allowed formats: PDF, PNG, JPG, JPEG."
            ),
        )

    # Validate extraction mode
    extraction_mode = extraction_mode.strip().lower()

    if extraction_mode not in ["agentic", "nonagentic"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid extraction_mode. Use agentic or nonagentic.",
        )

    # Read uploaded file
    try:

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 25 MB limit.",
            )

    finally:
        await file.close()

    # Generate unique job ID
    job_id = uuid.uuid4().hex

    # Preserve the original extension
    input_path = INPUT_DIR / f"{job_id}{file_ext}"

    # Save original uploaded file
    input_path.write_bytes(content)

    logger.info("----------------------------------------")
    logger.info("New extraction request")
    logger.info("Job ID: %s", job_id)
    logger.info("Original filename: %s", original_filename)
    logger.info("Extraction mode: %s", extraction_mode)
    logger.info("Saved input path: %s", input_path)
    logger.info("Input extension: %s", input_path.suffix.lower())
    logger.info("----------------------------------------")

    try:

        # ------------------------------------------
        # AGENTIC MODE
        # ------------------------------------------

        if extraction_mode == "agentic":

            logger.info("Processing using AGENTIC mode")

            # Agentic workflow expects a PDF.
            # Convert images to PDF only in this mode.

            if file_ext == ".pdf":

                processing_path = input_path

            else:

                processing_path = INPUT_DIR / f"{job_id}.pdf"

                await run_in_threadpool(
                    convert_image_to_pdf,
                    input_path,
                    processing_path,
                )

            result = await run_in_threadpool(
                process_agentic_document,
                processing_path,
                job_id,
                user_query,
            )

        # ------------------------------------------
        # NON-AGENTIC MODE
        # ------------------------------------------

        else:

            logger.info("Processing using NON-AGENTIC mode")

            # Pass the original file directly.
            # PNG/JPG/JPEG must NOT be passed to PDF readers.

            result = await run_in_threadpool(
                process_nonagentic_document,
                input_path,
                job_id,
            )

        logger.info(
            "Extraction completed successfully. Job ID: %s",
            job_id,
        )

        return {
            "status": "success",
            "message": "Document extracted successfully.",
            "job_id": job_id,
            "extraction_mode": extraction_mode,
            "document_type": result["document_type"],
            "extraction_method": result["extraction_method"],
            "extracted_data": result["extracted_data"],
            "excel_file": result["excel_file"],
            "download_url": (
                f"/api/v1/documents/{job_id}/download"
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Document extraction failed. Job ID: %s",
            job_id,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document extraction failed: {str(exc)}",
        ) from exc


# --------------------------------------------------
# DOWNLOAD EXCEL ENDPOINT
# --------------------------------------------------

@app.get("/api/v1/documents/{job_id}/download")
async def download_document(

    job_id: str,

    authenticated: bool = Depends(verify_api_key),

):

    # Validate job ID
    try:

        safe_job_id = uuid.UUID(hex=job_id).hex

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Invalid job ID.",
        )

    job_folder = OUTPUT_DIR / safe_job_id

    if not job_folder.exists():
        raise HTTPException(
            status_code=404,
            detail="Document job not found.",
        )

    # Find generated Excel file
    excel_files = list(
        job_folder.rglob("*.xlsx")
    )

    if not excel_files:
        raise HTTPException(
            status_code=404,
            detail="Excel output file not found.",
        )

    excel_path = excel_files[0]

    return FileResponse(
        path=str(excel_path),
        filename=excel_path.name,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# --------------------------------------------------
# RUN LOCALLY
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )


import logging
import uuid
import os
import secrets

from io import BytesIO
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from PIL import Image, ImageSequence, ImageOps

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

from graph_workflow_updated import create_document_router_graph
from main_updated import save_result_to_excel


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_KEY = os.getenv("API_KEY")

INPUT_DIR = BASE_DIR / "input_documents"
OUTPUT_DIR = BASE_DIR / "output_results"

MAX_FILE_SIZE = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# API KEY AUTHENTICATION
# --------------------------------------------------

async def verify_api_key(
    x_api_key: Optional[str] = Header(default=None)
):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured."
        )

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key is required."
        )

    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key."
        )

    return True


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Document Extractor API",
    description="API for extracting information from PDF and image documents.",
    version="5.0.0"
)


# --------------------------------------------------
# IMAGE TO PDF CONVERSION
# --------------------------------------------------

def convert_image_to_pdf(content: bytes, output_path: Path):

    try:
        frames = []

        with Image.open(BytesIO(content)) as image:

            for frame in ImageSequence.Iterator(image):

                # Correct image orientation
                frame = ImageOps.exif_transpose(frame.copy())

                # Handle transparent images with a white background
                if frame.mode in ("RGBA", "LA") or (
                    frame.mode == "P" and "transparency" in frame.info
                ):
                    rgba = frame.convert("RGBA")
                    background = Image.new(
                        "RGBA",
                        rgba.size,
                        (255, 255, 255, 255)
                    )
                    frame = Image.alpha_composite(
                        background,
                        rgba
                    ).convert("RGB")
                else:
                    frame = frame.convert("RGB")

                frames.append(frame.copy())

        if not frames:
            raise ValueError("No valid image pages found.")

        # Convert image frames into a PDF
        frames[0].save(
            output_path,
            format="PDF",
            save_all=True,
            append_images=frames[1:]
        )

        # Verify generated PDF
        if not output_path.exists():
            raise ValueError("PDF file was not generated.")

        if not output_path.read_bytes().startswith(b"%PDF-"):
            raise ValueError("Generated file is not a valid PDF.")

        logger.info("Image successfully converted to PDF: %s", output_path)

    except Exception as exc:
        logger.exception("Image to PDF conversion failed")

        raise ValueError(
            f"Unable to convert image to PDF: {str(exc)}"
        ) from exc


# --------------------------------------------------
# AGENTIC DOCUMENT PROCESSING
# --------------------------------------------------

def process_agentic_document(
    pdf_path: Path,
    job_id: str,
    user_query: Optional[str] = None
):

    logger.info("Starting AGENTIC extraction")

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
            "Agentic workflow returned no structured data."
        )

    output_folder = OUTPUT_DIR / job_id
    output_folder.mkdir(parents=True, exist_ok=True)

    save_result_to_excel(
        extracted_data=extracted_data,
        output_dir=str(output_folder),
        base_filename=pdf_path.stem
    )

    excel_path = output_folder / f"{pdf_path.stem}_extracted.xlsx"

    if not excel_path.is_file():
        raise FileNotFoundError(
            "Agentic Excel output was not created."
        )

    return {
        "document_type": result.get("document_type"),
        "extraction_method": result.get("extraction_method"),
        "extracted_data": extracted_data.model_dump(),
        "excel_file": excel_path.name,
    }


# --------------------------------------------------
# NON-AGENTIC AWS TEXTRACT PROCESSING
# --------------------------------------------------

def process_nonagentic_document(
    pdf_path: Path,
    job_id: str
):

    logger.info("Starting NON-AGENTIC AWS Textract extraction")

    from AwsTextract_Hybrid import (
        process_file,
        save_invoice_to_excel
    )

    textract_result = process_file(str(pdf_path))

    if not textract_result:
        raise ValueError(
            "AWS Textract returned no extraction result."
        )

    output_folder = OUTPUT_DIR / job_id
    output_folder.mkdir(parents=True, exist_ok=True)

    save_invoice_to_excel(
        filename=pdf_path.name,
        result=textract_result,
        output_folder_path=str(output_folder)
    )

    excel_path = (
        output_folder
        / pdf_path.stem
        / "extracted_invoice_data.xlsx"
    )

    if not excel_path.is_file():
        raise FileNotFoundError(
            "AWS Textract Excel output was not created."
        )

    return {
        "document_type": "INVOICE",
        "extraction_method": "AWS_TEXTRACT",
        "extracted_data": textract_result,
        "excel_file": excel_path.name,
    }


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
async def root():

    return {
        "message": "Document Extractor API is running",
        "version": "5.0.0",
        "supported_files": list(ALLOWED_EXTENSIONS),
        "extraction_modes": [
            "agentic",
            "nonagentic"
        ],
        "endpoints": {
            "health": "/health",
            "extract": "/api/v1/documents/extract",
            "download": "/api/v1/documents/{job_id}/download",
            "docs": "/docs"
        },
        "authentication": "X-API-Key required for extraction and download"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "message": "Document Extractor API is running"
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
        file.filename or "document"
    ).name

    file_extension = Path(
        original_filename
    ).suffix.lower()

    # Validate file extension
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Allowed formats: PDF, JPG, JPEG, PNG, TIFF, WEBP."
            )
        )

    # Validate extraction mode
    extraction_mode = extraction_mode.strip().lower()

    if extraction_mode not in ["agentic", "nonagentic"]:
        raise HTTPException(
            status_code=400,
            detail="Choose agentic or nonagentic."
        )

    # Read uploaded file
    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Maximum file size is 25 MB."
            )

    finally:
        await file.close()

    # Create unique job ID
    job_id = uuid.uuid4().hex

    # All input files are normalized to PDF
    pdf_path = INPUT_DIR / f"{job_id}.pdf"

    try:

        # ------------------------------------------
        # PDF FILE
        # ------------------------------------------

        if file_extension == ".pdf":

            if not content.startswith(b"%PDF-"):
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is not a valid PDF."
                )

            pdf_path.write_bytes(content)

            logger.info(
                "PDF uploaded successfully: %s",
                original_filename
            )

        # ------------------------------------------
        # IMAGE FILE
        # ------------------------------------------

        else:

            convert_image_to_pdf(
                content,
                pdf_path
            )

            logger.info(
                "Image converted to PDF: %s",
                original_filename
            )

        # ------------------------------------------
        # EXTRACTION
        # ------------------------------------------

        logger.info(
            "Starting extraction. Job ID: %s, Mode: %s",
            job_id,
            extraction_mode
        )

        if extraction_mode == "agentic":

            result = await run_in_threadpool(
                process_agentic_document,
                pdf_path,
                job_id,
                user_query
            )

        else:

            result = await run_in_threadpool(
                process_nonagentic_document,
                pdf_path,
                job_id
            )

        logger.info(
            "Extraction completed successfully. Job ID: %s",
            job_id
        )

        return {
            "status": "success",
            "message": "Document extracted successfully.",
            "job_id": job_id,
            "original_filename": original_filename,
            "file_type": file_extension.replace(".", ""),
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
            "Extraction failed for job %s",
            job_id
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document extraction failed: {str(exc)}"
        ) from exc


# --------------------------------------------------
# DOWNLOAD EXCEL ENDPOINT
# --------------------------------------------------

@app.get("/api/v1/documents/{job_id}/download")
async def download_excel(

    job_id: str,

    authenticated: bool = Depends(verify_api_key),

):

    # Validate job ID
    try:
        valid_job_id = uuid.UUID(hex=job_id).hex

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid job ID."
        )

    output_folder = OUTPUT_DIR / valid_job_id

    if not output_folder.exists():
        raise HTTPException(
            status_code=404,
            detail="Document job not found."
        )

    # Find generated Excel file
    excel_files = list(
        output_folder.rglob("*.xlsx")
    )

    if not excel_files:
        raise HTTPException(
            status_code=404,
            detail="Excel file not found."
        )

    excel_path = excel_files[0]

    return FileResponse(
        path=excel_path,
        filename=excel_path.name,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

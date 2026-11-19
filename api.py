
import logging
import uuid
import os
import secrets
import threading

from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

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

from graph_workflow_updated import create_document_router_graph
from main_updated import save_result_to_excel


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

API_KEY = os.getenv("API_KEY")

INPUT_DIR = BASE_DIR / "input_documents"
OUTPUT_DIR = BASE_DIR / "output_results"

MAX_FILE_SIZE = 25 * 1024 * 1024

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Background job configuration
executor = ThreadPoolExecutor(max_workers=2)

jobs = {}
jobs_lock = threading.Lock()


# --------------------------------------------------
# 2. API KEY AUTHENTICATION
# --------------------------------------------------

async def verify_api_key(
    x_api_key: Optional[str] = Header(default=None)
):

    if not API_KEY:
        logger.error("API_KEY is not configured")

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
# 3. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Document Extractor API",
    description=(
        "Upload a PDF and choose either agentic LangGraph "
        "extraction or non-agentic AWS Textract extraction. "
        "Supports asynchronous background processing."
    ),
    version="3.0.0",
)


# --------------------------------------------------
# 4. AGENTIC EXTRACTION
# --------------------------------------------------

def process_agentic_document(
    pdf_path: Path,
    job_id: str,
    user_query: Optional[str] = None,
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
        base_filename=pdf_path.stem,
    )

    excel_path = (
        output_folder /
        f"{pdf_path.stem}_extracted.xlsx"
    )

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
# 5. NON-AGENTIC AWS TEXTRACT EXTRACTION
# --------------------------------------------------

def process_nonagentic_document(
    pdf_path: Path,
    job_id: str,
):

    logger.info(
        "Starting NON-AGENTIC AWS Textract extraction"
    )

    from AwsTextract_Hybrid import (
        process_file,
        save_invoice_to_excel,
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
        output_folder_path=str(output_folder),
    )

    excel_path = (
        output_folder /
        pdf_path.stem /
        "extracted_invoice_data.xlsx"
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
# 6. BACKGROUND JOB PROCESSOR
# --------------------------------------------------

def process_document_in_background(
    job_id: str,
    pdf_path: Path,
    extraction_mode: str,
    user_query: Optional[str] = None,
):

    logger.info(
        "Background processing started for job %s",
        job_id,
    )

    with jobs_lock:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["started_at"] = datetime.now(
            timezone.utc
        ).isoformat()

    try:

        if extraction_mode == "agentic":

            logger.info(
                "Job %s routed to AGENTIC workflow",
                job_id,
            )

            result = process_agentic_document(
                pdf_path,
                job_id,
                user_query,
            )

        else:

            logger.info(
                "Job %s routed to NON-AGENTIC workflow",
                job_id,
            )

            result = process_nonagentic_document(
                pdf_path,
                job_id,
            )

        with jobs_lock:
            jobs[job_id].update({
                "status": "completed",
                "document_type": result["document_type"],
                "extraction_method": result["extraction_method"],
                "extracted_data": result["extracted_data"],
                "excel_file": result["excel_file"],
                "completed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            })

        logger.info(
            "Job %s completed successfully",
            job_id,
        )

    except Exception:

        logger.exception(
            "Background processing failed for job %s",
            job_id,
        )

        with jobs_lock:
            jobs[job_id].update({
                "status": "failed",
                "error": (
                    "Document processing failed. "
                    "Check the API logs."
                ),
                "completed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            })


# --------------------------------------------------
# 7. API HOMEPAGE (PUBLIC)
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Document Extractor API is running.",
        "available_modes": [
            "agentic",
            "nonagentic"
        ],
        "docs": "/docs",
        "health": "/health",
        "authentication": "API Key required for document APIs"
    }


# --------------------------------------------------
# 8. HEALTH CHECK (PUBLIC)
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# 9. ASYNCHRONOUS DOCUMENT EXTRACTION
# --------------------------------------------------

@app.post("/api/v1/documents/extract", status_code=202)
async def extract_document(
    file: UploadFile = File(...),

    extraction_mode: str = Form(...),

    user_query: Optional[str] = Form(default=None),

    authenticated: bool = Depends(verify_api_key),
):

    original_filename = Path(
        file.filename or "document.pdf"
    ).name

    extraction_mode = extraction_mode.strip().lower()

    # Validate extraction mode
    if extraction_mode not in ["agentic", "nonagentic"]:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid extraction_mode. "
                "Choose agentic or nonagentic."
            ),
        )

    # Validate file extension
    if Path(original_filename).suffix.lower() != ".pdf":

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Maximum PDF size is 25 MB.",
            )

        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid PDF.",
            )

    finally:
        await file.close()

    # Generate job ID
    job_id = uuid.uuid4().hex

    pdf_path = INPUT_DIR / f"{job_id}.pdf"

    pdf_path.write_bytes(content)

    # Initialize job
    with jobs_lock:

        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "extraction_mode": extraction_mode,
            "original_filename": original_filename,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    try:

        # Submit extraction to background executor
        executor.submit(
            process_document_in_background,
            job_id,
            pdf_path,
            extraction_mode,
            user_query,
        )

    except Exception as exc:

        logger.exception(
            "Unable to submit background job %s",
            job_id,
        )

        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Unable to start processing."

        raise HTTPException(
            status_code=500,
            detail="Unable to start document processing.",
        ) from exc

    logger.info(
        "Job %s accepted for background processing",
        job_id,
    )

    # Return immediately without waiting for extraction
    return {
        "status": "accepted",
        "message": "Document uploaded successfully. Processing started.",
        "job_id": job_id,
        "extraction_mode": extraction_mode,
        "status_url": (
            f"/api/v1/documents/{job_id}/status"
        ),
        "download_url": (
            f"/api/v1/documents/{job_id}/download"
        ),
    }


# --------------------------------------------------
# 10. JOB STATUS ENDPOINT
# --------------------------------------------------

@app.get("/api/v1/documents/{job_id}/status")
def get_job_status(
    job_id: str,

    authenticated: bool = Depends(verify_api_key),
):

    try:
        safe_job_id = uuid.UUID(hex=job_id).hex

    except (ValueError, AttributeError):

        raise HTTPException(
            status_code=400,
            detail="Invalid job ID.",
        )

    with jobs_lock:

        job = jobs.get(safe_job_id)

        if job is None:

            raise HTTPException(
                status_code=404,
                detail="Job not found.",
            )

        response = dict(job)

    if response["status"] == "completed":

        response["download_url"] = (
            f"/api/v1/documents/{safe_job_id}/download"
        )

    return response


# --------------------------------------------------
# 11. EXCEL DOWNLOAD ENDPOINT
# --------------------------------------------------

@app.get("/api/v1/documents/{job_id}/download")
def download_excel(
    job_id: str,

    authenticated: bool = Depends(verify_api_key),
):

    try:

        safe_job_id = uuid.UUID(hex=job_id).hex

    except (ValueError, AttributeError):

        raise HTTPException(
            status_code=400,
            detail="Invalid job ID.",
        )

    with jobs_lock:

        job = jobs.get(safe_job_id)

        if job is None:

            raise HTTPException(
                status_code=404,
                detail="Job not found.",
            )

        if job["status"] != "completed":

            raise HTTPException(
                status_code=409,
                detail=(
                    "Document processing is not completed yet. "
                    "Check the job status endpoint."
                ),
            )

    output_folder = OUTPUT_DIR / safe_job_id

    excel_files = list(
        output_folder.rglob("*.xlsx")
    )

    if not excel_files:

        raise HTTPException(
            status_code=404,
            detail="Excel file was not found for this job.",
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

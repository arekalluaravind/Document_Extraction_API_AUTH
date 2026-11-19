import logging
import uuid
import os
import secrets




from pathlib import Path
from typing import Optional




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
from starlette.concurrency import run_in_threadpool




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








# --------------------------------------------------
# 2. API KEY AUTHENTICATION
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
# 3. FASTAPI APPLICATION
# --------------------------------------------------




app = FastAPI(
    title="Document Extractor API",
    description=(
        "Upload a PDF or Image and receive complete invoice extraction data. "
        "Download the generated Excel file using the job ID."
    ),
    version="4.0.0",
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
# 6. API HOMEPAGE
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
# 7. HEALTH CHECK
# --------------------------------------------------




@app.get("/health")
def health():




    return {
        "status": "healthy"
    }








# --------------------------------------------------
# 8. API 1 - INVOICE EXTRACTION
# --------------------------------------------------




@app.post("/api/v1/documents/extract")
async def extract_document(
    file: UploadFile = File(...),




    extraction_mode: str = Form(...),




    user_query: Optional[str] = Form(default=None),




    authenticated: bool = Depends(verify_api_key),
):




    extraction_mode = extraction_mode.strip().lower()




    if extraction_mode not in ["agentic", "nonagentic"]:
        raise HTTPException(
            status_code=400,
            detail="Choose agentic or nonagentic."
        )




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




    if content.startswith(b"%PDF-"):
        file_ext = ".pdf"
    elif content.startswith(b"\x89PNG\r\n\x1a\n"):
        file_ext = ".png"
    elif content.startswith(b"\xff\xd8\xff"):
        file_ext = ".jpg"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload PDF, PNG, or JPG."
        )




    job_id = uuid.uuid4().hex




    pdf_path = INPUT_DIR / f"{job_id}{file_ext}"




    pdf_path.write_bytes(content)




    logger.info(
        "Starting extraction for job %s (detected %s)",
        job_id,
        file_ext,
    )




    try:




        if extraction_mode == "agentic":




            result = await run_in_threadpool(
                process_agentic_document,
                pdf_path,
                job_id,
                user_query,
            )




        else:




            result = await run_in_threadpool(
                process_nonagentic_document,
                pdf_path,
                job_id,
            )




        logger.info(
            "Extraction completed for job %s",
            job_id
        )




        return {
            "status": "success",
            "message": "Invoice extracted successfully.",
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
# 9. API 2 - EXCEL DOWNLOAD
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
            detail="Invalid job ID."
        )




    output_folder = OUTPUT_DIR / safe_job_id




    if not output_folder.exists():
        raise HTTPException(
            status_code=404,
            detail="Excel file not found for this job."
        )




    excel_files = list(
        output_folder.rglob("*.xlsx")
    )




    if not excel_files:
        raise HTTPException(
            status_code=404,
            detail="Excel file was not found."
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

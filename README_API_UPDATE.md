# Document Extractor API update

## Files to copy into your existing project

- Replace `api.py` with the included `api.py`.
- Replace `graph_workflow_updated.py` with the included updated file.
- Replace `models/schemas.py` with the included updated schema.
- Keep your existing `main_updated.py` unchanged.
- Keep `.env` in your project root and ensure it contains a valid `OPENROUTER_API_KEY`.
- Ensure `models/__init__.py` exists (it can be empty).

## Expected project layout

DocumentExtractor/
  .venv/
  input_documents/
  output_results/
  models/
    __init__.py
    schemas.py
  .env
  api.py
  graph_workflow_updated.py
  main_updated.py

## Install missing API dependencies

In the PyCharm terminal with the project virtual environment activated:

    pip install fastapi "uvicorn[standard]" python-multipart python-dotenv

Your existing dependencies must also be installed:

    pip install langgraph openai pandas openpyxl pdfplumber pytesseract pdf2image pypdf

Scanned PDFs on Windows may require Poppler and Tesseract installed and available on PATH.

## Start the API

    uvicorn api:app --host 127.0.0.1 --port 8000 --reload

Open:

    http://127.0.0.1:8000/docs

## Endpoints

- GET `/` - API information
- GET `/health` - health check
- POST `/api/v1/documents/extract` - upload one PDF (multipart field `file`; optional form field `user_query`)
- GET `/api/v1/documents/{job_id}/download` - download the generated Excel workbook

The API saves uploaded PDFs to `input_documents/` and Excel files to
`output_results/<job_id>/`.

## Notes

This keeps the existing OpenRouter/NVIDIA Nemotron provider configuration.
It does not switch the project to Gemini or AWS Textract.
Before exposing the API outside your machine, add authentication, rate limiting,
upload scanning, HTTPS, and production deployment controls.

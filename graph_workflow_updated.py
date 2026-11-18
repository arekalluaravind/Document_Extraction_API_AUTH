import os
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
import pdfplumber
import pytesseract
from langgraph.graph import END, START, StateGraph

from models.schemas import (
    ClassificationResult,
    ContractQueryResult,
    InvoiceData,
    PurchaseOrderData,
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing. Add it to your local .env file."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b:free"


class DocumentState(TypedDict):
    pdf_path: str
    user_query: Optional[str]
    extracted_text: str
    extraction_method: str
    uploaded_file: Optional[Any]
    document_type: Optional[str]
    extracted_data: Optional[Any]


def call_llm(prompt_text: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert document extraction and classification "
                    "assistant. Return only valid JSON with all requested keys. "
                    "Do not invent information. Use null for unavailable nullable "
                    "values, empty arrays for missing lists, and empty strings "
                    "for unavailable required text fields."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("The language model returned an empty response.")
    return content


def ingest_and_extract_node(state: DocumentState) -> Dict[str, Any]:
    pdf_path = state["pdf_path"]
    print(f"[Smart Ingestion] Inspecting: {pdf_path}")

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    extracted_text = ""
    extraction_method = "pdfplumber (selectable)"

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
    except Exception:
        # OCR fallback below will be attempted if no usable text was extracted.
        pass

    if len(extracted_text.strip()) < 50:
        print("[OCR Mode] Running Tesseract OCR...")
        extraction_method = "pytesseract (scanned/OCR)"
        images = convert_from_path(pdf_path)
        ocr_pages = []
        for page_number, image in enumerate(images, start=1):
            ocr_pages.append(
                f"--- Page {page_number} ---\n{pytesseract.image_to_string(image)}"
            )
        extracted_text = "\n".join(ocr_pages)

    if not extracted_text.strip():
        raise ValueError("No readable text could be extracted from the PDF.")

    return {
        "extracted_text": extracted_text,
        "extraction_method": extraction_method,
    }


def keyword_classifier(text: str) -> Optional[str]:
    text_lower = text.lower()

    keyword_groups = {
        "invoice": ["invoice", "bill to", "tax invoice", "amount due", "invoice no"],
        "purchase_order": ["purchase order", "po number", "ship to", "buyer", "vendor code"],
        "contract_agreement": [
            "agreement", "contract", "parties", "hereby",
            "terms and conditions", "effective date",
        ],
    }

    scores = {
        name: sum(1 for keyword in keywords if keyword in text_lower)
        for name, keywords in keyword_groups.items()
    }

    highest_score = max(scores.values())
    if highest_score == 0:
        return None

    winners = [name for name, score in scores.items() if score == highest_score]
    return winners[0] if len(winners) == 1 else None


def orchestrator_classifier_node(state: DocumentState) -> Dict[str, Any]:
    print("[Orchestrator Agent] Classifying document...")
    extracted_text = state["extracted_text"]

    document_type = keyword_classifier(extracted_text)
    if document_type:
        print(f"[Keyword Match] {document_type}")
        return {"document_type": document_type}

    print("[LLM Classifier] Calling OpenRouter...")
    prompt = f"""
Classify this document into exactly one category:
- invoice
- purchase_order
- contract_agreement

Return a JSON object containing BOTH keys:
{{
  "document_type": "invoice",
  "confidence": 0.95
}}

Rules:
- document_type must be exactly one of the three categories above.
- confidence must be a number from 0.0 to 1.0.
- Do not include any other keys.

Document text:
{extracted_text[:5000]}
"""
    classification = ClassificationResult.model_validate_json(call_llm(prompt))
    document_type = classification.document_type.lower()

    if document_type not in {"invoice", "purchase_order", "contract_agreement"}:
        raise ValueError(f"Unsupported document type returned: {document_type}")

    print(
        f"[Classification Result] {document_type}; "
        f"confidence={classification.confidence}"
    )
    return {"document_type": document_type}


def invoice_extractor_node(state: DocumentState) -> Dict[str, Any]:
    print("[Invoice Agent] Extracting invoice data...")
    user_query = state.get("user_query")
    query_instruction = (
        f"\nUser's specific request: {user_query}" if user_query else ""
    )

    prompt = f"""
Extract the invoice into one JSON object with exactly these fields:
{{
  "targeted_answer": null,
  "invoice_number": "",
  "vendor_name": "",
  "customer_name": null,
  "invoice_date": "",
  "due_date": null,
  "line_items": [
    {{
      "description": "",
      "quantity": 0.0,
      "unit_price": 0.0,
      "total_price": 0.0
    }}
  ],
  "total_amount": 0.0
}}

Rules:
- Use numbers for quantity, unit_price, total_price and total_amount.
- Use null for customer_name or due_date if not present.
- Use an empty list if no line items are present.
- Do not guess missing values.
- targeted_answer may be null when no specific request is provided.
- Do not include fields outside this schema.

Document text:
{state["extracted_text"]}
{query_instruction}
"""
    data = InvoiceData.model_validate_json(call_llm(prompt))
    return {"extracted_data": data}


def purchase_order_extractor_node(state: DocumentState) -> Dict[str, Any]:
    print("[Purchase Order Agent] Extracting PO data...")
    user_query = state.get("user_query")
    query_instruction = (
        f"\nUser's specific request: {user_query}" if user_query else ""
    )

    prompt = f"""
Extract the purchase order into one JSON object with exactly these fields:
{{
  "targeted_answer": null,
  "po_number": "",
  "buyer_company": "",
  "supplier_name": "",
  "order_date": "",
  "line_items": [
    {{
      "description": "",
      "quantity": 0.0,
      "unit_price": 0.0,
      "total_price": 0.0
    }}
  ],
  "total_order_value": 0.0
}}

Rules:
- Use numeric values for quantity, unit_price, total_price and total_order_value.
- Use an empty list if there are no line items.
- Do not guess missing values; use empty strings or 0.0 where required.
- targeted_answer may be null if no specific request was supplied.
- Do not include fields outside this schema.

Document text:
{state["extracted_text"]}
{query_instruction}
"""
    data = PurchaseOrderData.model_validate_json(call_llm(prompt))
    return {"extracted_data": data}


def contract_analyzer_node(state: DocumentState) -> Dict[str, Any]:
    print("[Contract Agent] Analyzing contract...")
    user_query = state.get("user_query")
    query_instruction = (
        f"\nUser's specific question: {user_query}" if user_query else ""
    )

    prompt = f"""
Analyze the contract and return one JSON object with exactly these fields:
{{
  "targeted_answer": null,
  "document_type": "Contract Agreement",
  "summary": "",
  "referenced_sections": []
}}

Rules:
- summary must describe the main terms, obligations and relevant conditions.
- referenced_sections must be an array of section names/numbers or clause references.
- targeted_answer may be null when no specific question was supplied.
- Do not invent clauses or section references.
- Do not include fields outside this schema.

Document text:
{state["extracted_text"]}
{query_instruction}
"""
    data = ContractQueryResult.model_validate_json(call_llm(prompt))
    return {"extracted_data": data}


def route_to_specialist(state: DocumentState) -> str:
    routes = {
        "invoice": "invoice_agent",
        "purchase_order": "purchase_order_agent",
        "contract_agreement": "contract_agent",
    }
    document_type = state.get("document_type")
    if document_type not in routes:
        raise ValueError(f"Unsupported document type: {document_type}")
    return routes[document_type]


def create_document_router_graph():
    workflow = StateGraph(DocumentState)

    workflow.add_node("ingest", ingest_and_extract_node)
    workflow.add_node("orchestrator", orchestrator_classifier_node)
    workflow.add_node("invoice_agent", invoice_extractor_node)
    workflow.add_node("purchase_order_agent", purchase_order_extractor_node)
    workflow.add_node("contract_agent", contract_analyzer_node)

    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_to_specialist,
        {
            "invoice_agent": "invoice_agent",
            "purchase_order_agent": "purchase_order_agent",
            "contract_agent": "contract_agent",
        },
    )

    workflow.add_edge("invoice_agent", END)
    workflow.add_edge("purchase_order_agent", END)
    workflow.add_edge("contract_agent", END)

    return workflow.compile()

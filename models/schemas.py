from typing import List, Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(
        default="",
        description="Description of the item or service",
    )
    quantity: float = Field(
        default=0.0,
        description="Quantity ordered or billed",
    )
    unit_price: float = Field(
        default=0.0,
        description="Price per single unit",
    )
    total_price: float = Field(
        default=0.0,
        description="Total price for the line item",
    )


class InvoiceData(BaseModel):
    targeted_answer: Optional[str] = Field(
        default=None,
        description="Direct answer to the user's specific query, if provided.",
    )
    invoice_number: str = Field(
        default="",
        description="Invoice identification number",
    )
    vendor_name: str = Field(
        default="",
        description="Name of the supplier or vendor",
    )
    customer_name: Optional[str] = Field(
        default=None,
        description="Name of the customer or billed party, if available",
    )
    invoice_date: str = Field(
        default="",
        description="Date the invoice was issued",
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Payment due date, if available",
    )
    line_items: List[LineItem] = Field(
        default_factory=list,
        description="Tabular list of billed items",
    )
    total_amount: float = Field(
        default=0.0,
        description="Grand total of the invoice",
    )


class PurchaseOrderData(BaseModel):
    targeted_answer: Optional[str] = Field(
        default=None,
        description="Direct answer to the user's specific query, if provided.",
    )
    po_number: str = Field(default="", description="Purchase Order number")
    buyer_company: str = Field(
        default="",
        description="Company issuing the purchase order",
    )
    supplier_name: str = Field(default="", description="Target supplier name")
    order_date: str = Field(default="", description="Date of the purchase order")
    line_items: List[LineItem] = Field(
        default_factory=list,
        description="Tabular list of ordered items",
    )
    total_order_value: float = Field(default=0.0, description="Total PO amount")


class ContractQueryResult(BaseModel):
    targeted_answer: Optional[str] = Field(
        default=None,
        description="Direct answer or clause extracted based on the user's query.",
    )
    document_type: str = Field(default="Contract Agreement")
    summary: str = Field(
        default="",
        description="Comprehensive summary of terms and obligations",
    )
    referenced_sections: List[str] = Field(
        default_factory=list,
        description="Important sections or clauses found in the document",
    )


class ClassificationResult(BaseModel):
    document_type: str = Field(
        description=(
            "Must be precisely one of: invoice, purchase_order, "
            "or contract_agreement"
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )

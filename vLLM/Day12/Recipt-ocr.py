import base64
import json
from pydantic import BaseModel
from typing import Optional, List
from openai import OpenAI

class LineItem(BaseModel):
    name: str
    quantity: int
    unit_price: float
    total: float

class Receipt(BaseModel):
    store_name: str
    date: str
    items: List[LineItem]
    subtotal: float
    tax: float
    total: float
    payment_method: Optional[str] = None

def encode_image_url(image_path):
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_receipt(image_path_or_url, port=8000):
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
    url = image_path_or_url if image_path_or_url.startswith("http") else encode_image_url(image_path_or_url)
    response = client.chat.completions.create(
        model="local",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",      "text": "Extract all receipt data as structured JSON. Be precise with numbers."},
                {"type": "image_url", "image_url": {"url": url}},
            ]
        }],
        extra_body={"guided_json": Receipt.model_json_schema()},
        temperature=0.0,
        max_tokens=600,
    )
    return Receipt.model_validate_json(response.choices[0].message.content)

def receipt_summary(receipt):
    return (f"Store: {receipt.store_name} | Date: {receipt.date} | "
            f"Items: {len(receipt.items)} | Total: ${receipt.total:.2f}")

def validate_receipt(receipt):
    warnings = []
    items_sum = sum(item.total for item in receipt.items)
    if abs(items_sum - receipt.subtotal) > 0.01:
        warnings.append(f"Items sum ${items_sum:.2f} != subtotal ${receipt.subtotal:.2f}")
    if abs(receipt.subtotal + receipt.tax - receipt.total) > 0.01:
        warnings.append(f"subtotal + tax ${receipt.subtotal+receipt.tax:.2f} != total ${receipt.total:.2f}")
    for item in receipt.items:
        expected = item.quantity * item.unit_price
        if abs(expected - item.total) > 0.01:
            warnings.append(f"'{item.name}': {item.quantity} x ${item.unit_price:.2f} = ${expected:.2f} but total shows ${item.total:.2f}")
    return warnings


mock_json = """
{
  "store_name": "Whole Foods Market",
  "date": "2025-03-15",
  "items": [
    {"name": "Organic Milk",    "quantity": 2, "unit_price": 4.99, "total": 9.98},
    {"name": "Sourdough Bread", "quantity": 1, "unit_price": 6.50, "total": 6.50},
    {"name": "Avocados",        "quantity": 3, "unit_price": 1.50, "total": 4.60}
  ],
  "subtotal": 21.08,
  "tax": 1.69,
  "total": 22.77,
  "payment_method": "Visa"
}
"""

receipt = Receipt.model_validate_json(mock_json)
print(receipt_summary(receipt))
warnings = validate_receipt(receipt)
if warnings:
    for w in warnings: print(f"  WARN: {w}")
else:
    print("  All totals verified OK")
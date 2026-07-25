from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Order(BaseModel):
    order_id: str
    customer_name: str
    status: Literal["paid", "processing", "shipped", "delayed", "cancelled"]
    item: str
    amount_cny: float
    updated_at: datetime


class CreateTicketRequest(BaseModel):
    order_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    reason: str = Field(min_length=3, max_length=500)
    priority: Literal["low", "normal", "high"] = "normal"


class Ticket(BaseModel):
    ticket_id: str
    order_id: str
    reason: str
    priority: Literal["low", "normal", "high"]
    status: Literal["open", "closed"] = "open"
    created_at: datetime

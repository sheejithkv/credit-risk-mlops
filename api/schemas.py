from __future__ import annotations

from pydantic import BaseModel, Field


class CreditRiskRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    duration_months: int = Field(gt=0)
    credit_amount: int = Field(gt=0)
    installment_rate: int = Field(ge=1, le=4)
    existing_credits: int = Field(ge=0)
    people_liable: int = Field(ge=0)
    purpose: str
    checking_account: str
    savings: str
    employment: str
    personal_status: str
    housing: str
    job: str
    property: str
    other_installment_plans: str
    foreign_worker: str
    telephone: str
    credit_history: str


class CreditRiskResponse(BaseModel):
    prediction: int
    risk_label: str
    probability_bad: float | None

# Created by aaronkueh on 9/19/2025
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime, date

class TSQueryInput(BaseModel):
    asset_id: str
    signals: List[str]
    start: datetime
    end: datetime
    resample: Optional[str] = Field(default=None, description="e.g., '5min'")
    features: Optional[List[Literal["rms","ewma","kurtosis","skew","var","slope"]]] = None

class AnomalyResult(BaseModel):
    asset_id: str
    signal: str
    type: Literal["trend","baseline_shift","spike","variance"]
    score: float
    window_from: datetime
    window_to: datetime
    evidence: dict

class RULInput(BaseModel):
    asset_id: str
    component_id: str
    last_service: date
    features_hint: Optional[List[str]] = None  # e.g., ["vibration_rms","temp"]

class RULEstimate(BaseModel):
    asset_id: str
    component_id: str
    rul_days: float
    confidence: float
    model_ver: str
    asof: datetime

class ScheduleInput(BaseModel):
    asset_id: str
    oem_cycle_days: int
    last_service: date
    rul_days: Optional[float] = None
    criticality: Optional[int] = 3
    horizon_days: int = 14

class ScheduleDecision(BaseModel):
    target_date: date
    reason: str

class InventoryCheckInput(BaseModel):
    task_list_id: str
    reservation_qty: Optional[dict] = None  # {part_no: qty}

class InventoryCheckResult(BaseModel):
    ok: bool
    missing: List[dict]  # [{"part_no":..., "need":..., "on_hand":..., "min":..., "lead_time_days":...}]

class WorkOrderInput(BaseModel):
    asset_id: str
    plan_date: date
    task_list_id: str
    parts: List[dict]
    crew_req: dict
    trigger: dict

class WorkOrderResult(BaseModel):
    wo_id: str
    status: Literal["draft","created"]
# Created by aaronkueh on 9/19/2025

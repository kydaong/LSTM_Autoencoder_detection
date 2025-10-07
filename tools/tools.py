# Created by aaronkueh on 9/19/2025
# tools.py
from langchain.tools import tool
from contracts import *

@tool("ts_query", args_schema=TSQueryInput, return_direct=False)
def ts_query_tool(args: TSQueryInput) -> dict:
    """Query time-series DB and optionally compute features."""
    # call InfluxDB/Timescale; compute features server-side or here
    return {"data": "..."}  # keep raw/feature data; large payloads -> store and return URI

@tool("sql_query_ro")
def sql_query_ro_tool(query: str) -> str:
    """Readonly SQL query over AOM-Dev. Input must be a single SELECT."""
    # run via SQLAlchemy connection with allowlist; enforce SELECT-only
    return "json_rows_or_table"

@tool("rag_search")
def rag_search_tool(q: str) -> dict:
    """Hybrid search over OEM docs/SOPs with citations."""
    # qdrant + BM25 + rerank -> return top_k chunks + cites
    return {"hits": []}

@tool("pdf_summarize")
def pdf_summarize_tool(uri: str, pages: str = "all") -> dict:
    """Extract + summarize PDF (OCR if needed)."""
    return {"summary": "...", "citations": [...]}

@tool("anomaly_infer", args_schema=TSQueryInput)
def anomaly_infer_tool(args: TSQueryInput) -> list[AnomalyResult]:
    """Detect trend/shift/spikes/variance on given signals."""
    # compute; return list of AnomalyResult dicts
    return []

@tool("rul_infer", args_schema=RULInput)
def rul_infer_tool(args: RULInput) -> RULEstimate:
    """Estimate RUL using LSTM autoencoder features."""
    return RULEstimate(asset_id=args.asset_id, component_id=args.component_id,
                       rul_days=28.5, confidence=0.78, model_ver="lstm_ae_v2.1", asof=datetime.utcnow())

@tool("schedule_optimize", args_schema=ScheduleInput)
def schedule_optimize_tool(args: ScheduleInput) -> ScheduleDecision:
    """Compute target maintenance date using OEM cycle + RUL."""
    # α by criticality (0.6..0.9). Example stub:
    alpha = {1:0.6,2:0.7,3:0.8,4:0.85,5:0.9}.get(args.criticality, 0.8)
    # produce target_date ... (stub)
    return ScheduleDecision(target_date=date.today(), reason="stub")

@tool("inventory_check", args_schema=InventoryCheckInput)
def inventory_check_tool(args: InventoryCheckInput) -> InventoryCheckResult:
    """Check stock vs task list and compute PR need."""
    return InventoryCheckResult(ok=True, missing=[])

@tool("create_workorder", args_schema=WorkOrderInput)
def create_workorder_tool(args: WorkOrderInput) -> WorkOrderResult:
    """Create a draft Work Order in dbo.work_order and dbo.work_order_part."""
    return WorkOrderResult(wo_id="WO-2025-0919-017", status="created")

@tool("raise_pr")
def raise_pr_tool(part_no: str, qty: float, reason: str) -> str:
    """Raise a Purchase Request for a part."""
    return "PR-2025-000123"

@tool("report_generate")
def report_generate_tool(title: str, sections_json: str) -> str:
    """Generate a compact HTML/PNG report and return a URI."""
    return "file://report/asset_360_MTR-001.html"

@tool("online_search")
def online_search_tool(q: str) -> dict:
    """Web search fallback."""
    return {"results": []}

@tool("sap_api")
def sap_api_tool(endpoint: str, payload_json: str) -> dict:
    """Call SAP/CMMS endpoint (e.g., get PR/PO status)."""
    return {"status": "ok"}

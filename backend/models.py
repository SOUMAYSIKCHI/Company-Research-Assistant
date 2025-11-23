from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any

ResearchDepth = Literal["quick_summary", "deep_research", "full_account_plan"]


class StartResearchRequest(BaseModel):
    company_name: str
    depth: ResearchDepth
    # Collected from UI / tools, used in prompts
    extra_instructions: Optional[str] = None


class ResearchStep(BaseModel):
    id: str
    label: str
    status: Literal["pending", "running", "complete", "skipped", "error"]
    detail: Optional[str] = None


class ResearchMetadata(BaseModel):
    generated_at: str
    depth: ResearchDepth
    sources_used: List[str] = []
    steps: List[ResearchStep] = []
    conflicts_detected: bool = False
    confidence_score: float = 0.0


class CompanySection(BaseModel):
    title: str
    content: str


class AccountPlan(BaseModel):
    company_name: str
    depth: ResearchDepth

    # Core narrative pieces
    overview: str
    company_profile: str
    market_analysis: str
    financial_highlights: str
    product_portfolio: str
    technology_stack: str
    competitors: str
    swot: str
    opportunities: str
    risks: str
    account_plan_30_60_90: str

    # For charts / stats later (pie, bar, etc.)
    # Example: [{"name": "Revenue (B USD)", "value": 96.7}, ...]
    kpi_summary: Optional[List[Dict[str, Any]]] = None

    # Optional structured sources from the LLM (RAG/Web snippets, etc.)
    sources: Optional[Dict[str, Any]] = None

    # Progress / pipeline metadata
    metadata: Optional[ResearchMetadata] = None

    # Flattened list for UI rendering (right column cards)
    sections: List[CompanySection]


class StartResearchResponse(BaseModel):
    conversation_id: str
    account_plan: AccountPlan


class ChatMessageRequest(BaseModel):
    conversation_id: str
    message: str


class ChatMessageResponse(BaseModel):
    reply: str


class FeedbackRequest(BaseModel):
    conversation_id: str
    overall_notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_summary: str

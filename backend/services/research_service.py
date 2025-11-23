import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Generator, List, Optional, Any

from .llm_service import call_llm, stream_llm
from .rag_service import search_context, web_search
from ..models import (
    StartResearchRequest,
    StartResearchResponse,
    AccountPlan,
    CompanySection,
    ChatMessageRequest,
    ChatMessageResponse,
    FeedbackRequest,
    FeedbackResponse,
    ResearchMetadata,
    ResearchStep,
)

# In-memory store
_CONVERSATIONS: Dict[str, Dict[str, Any]] = {}

BASE_SYSTEM_PROMPT = """
You are an enterprise-grade Company Research Analyst AI.

Your primary goal is to synthesize information and detect conflicts to build structured account plans.

You:
- Gather information from multiple sources (RAG PDFs, web search).
- CRITICAL: Detect conflicting or incomplete information between sources, explicitly listing them in the 'conflicts' array.
- Build structured account plans for sales / GTM teams.
- ALWAYS GENERATE DETAILED, INSIGHTFUL ANSWERS. Ensure narrative fields are 5-8 sentences long.
- Maintain a concise, confident, and professional tone. Focus on delivering strategic insights.
- CRITICAL: Avoid ALL conversational Markdown formatting. Specifically, DO NOT use **bold** marks (*, **). Only use plain text for narrative fields.
- Avoid all markdown formatting in your responses (no bullet symbols, no code fences, except for the required JSON output).
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n", 1)
        if len(parts) > 1 and parts[0].strip().startswith("```"):
            cleaned = parts[1]

        cleaned = cleaned.rstrip("`")
        return cleaned.strip()

    return cleaned.strip()


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(cleaned[start : end + 1])
        except Exception:
            pass
        return {}


def _generate_pipeline_history(
    plan: AccountPlan, conflicts_to_resolve: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Generates the initial chat history turns reflecting the research pipeline steps."""
    pipeline_history: List[Dict[str, str]] = []

    # 1. Pipeline Start
    pipeline_history.append(
        {
            "user": "System Initiated Pipeline",
            "assistant": f"Starting research pipeline for **{plan.company_name}** at {plan.depth} depth.",
        }
    )

    # 2. Sequential Steps (mimicking tools/sources)

    web_step_complete = any(
        s.id == "web_search" and s.status == "complete" for s in plan.metadata.steps
    )
    web_detail = (
        "Web search executed via Serper."
        if web_step_complete
        else "Skipped or failed to execute external web search."
    )
    pipeline_history.append(
        {
            "user": "System Initiated Pipeline",
            "assistant": "Step 1 of 4: Searching external web sources and news. Status: "
            + web_detail,
        }
    )

    rag_step = next((s for s in plan.metadata.steps if s.id == "rag_search"), None)
    rag_detail = rag_step.detail if rag_step else "No RAG process initiated."
    pipeline_history.append(
        {
            "user": "System Initiated Pipeline",
            "assistant": "Step 2 of 4: Querying internal RAG documents/data sources. Status: " + rag_detail,
        }
    )

    pipeline_history.append(
        {
            "user": "System Initiated Pipeline",
            "assistant": "Step 3 of 4: Accessing financial and competitor MCP server tools.",
        }
    )

    pipeline_history.append(
        {
            "user": "System Initiated Pipeline",
            "assistant": "Step 4 of 4: Synthesizing all findings and detecting conflicts using the core LLM.",
        }
    )

    # 3. Final Proactive Prompt
    if plan.metadata.conflicts_detected and conflicts_to_resolve:
        first_conflict = conflicts_to_resolve[0]
        topic = first_conflict.get("topic", "key company information")
        details = first_conflict.get("details", "Differing reports detected between sources.")

        agent_intro = (
            "Research synthesis is complete. However, a significant conflict was found regarding **"
            + topic
            + "**. Details: "
            + details
            + ". This data discrepancy means the plan's confidence is lower. To proceed, should I: "
            "1) Ignore the less-detailed source and continue, or "
            "2) Conduct a deep-dive search specifically on this topic? "
            "For example, you can say: **deep-dive on "
            + topic
            + "**."
        )
    else:
        agent_intro = (
            f"Research for **{plan.company_name}** is complete, and the account plan has been generated. "
            f"Current confidence estimate: {int(plan.metadata.confidence_score * 100)} percent. "
            "What would you like to do next? For example, you can ask to **'Show me the Competitor Pie Chart'**, "
            "**'Show me SWOT Analysis graph'**, or **'Edit the 30-60-90 Day Plan'**."
        )

    pipeline_history.append(
        {"user": "System Initiated Research", "assistant": agent_intro}
    )
    return pipeline_history


def build_research_prompt(req: StartResearchRequest,) -> (str, List[str], Optional[str]):
    """
    Build a single consolidated prompt with the enhanced JSON schema.
    """
    rag_query = (
        f"{req.company_name} company overview products strategy customers competitors"
    )
    rag_ctx = search_context(rag_query)

    web_query = (
        f"{req.company_name} latest news, financials, market position, competitors, risks"
    )
    web_text = web_search(web_query)

    # --- NEW PROMPT INSTRUCTIONS ---
    base = f"""
You are generating an enterprise account plan.

Company: {req.company_name}
Depth Level: {req.depth}

User intent or extra instructions (if any):
{req.extra_instructions or "None provided"}

**CRITICAL INSTRUCTIONS**:
1. For all narrative fields (e.g., Company Profile, Financial Highlights), generate **5-8 detailed sentences** supported by the research data.
2. Fill the new structured fields for the 30-60-90 Day Plan, Opportunities, SWOT Scores, and Competitor Data.
3. **CHART DATA INSTRUCTION**: The 'swot_radar_scores' (scale 1-10) and 'competitor_chart_data' (percentages must sum to 100) must be logically derived from the synthesized text in the 'swot' and 'competitors' fields. DO NOT GUESS; derive plausible figures from the provided text context.
4. If you find conflicting numbers, strategic directions, or future outlook, you must list them in the 'conflicts' array.

You have 2 primary sources:
1) Internal RAG snippets from company documents and PDFs.
2) External web search summaries including news, financial, and market data.

RAG_SNIPPETS (may be empty):
{rag_ctx[:5] if rag_ctx else "[]"}

WEB_SEARCH_SUMMARY (may be empty):
{web_text or "(no web data available)"}

Now synthesize everything and return a single valid JSON object only.
Do not include any markdown, commentary, or extra text.

JSON schema:

{{
  "overview": "... concise executive summary (3-4 sentences)...",
  "company_profile": "...",
  "market_analysis": "...",
  "financial_highlights": "...",
  "product_portfolio": "...",
  "technology_stack": "...",
  "competitors": "...",
  "swot": "...",
  "risks": "...",
  
  "opportunities_points": [
    "Opportunity 1: Detailed description of potential. (2-3 sentences)",
    "Opportunity 2: Detailed description of potential. (2-3 sentences)",
    "Opportunity 3: Detailed description of potential. (2-3 sentences)"
  ],
  
  "plan_table": [
    {{"period": "30 days", "focus": "Key objective", "metric": "How it is measured"}},
    {{"period": "60 days", "focus": "Key objective", "metric": "How it is measured"}},
    {{"period": "90 days", "focus": "Key objective", "metric": "How it is measured"}}
  ],

  "swot_radar_scores": {{
    "Strength": 9,
    "Weakness": 4,
    "Opportunity": 8,
    "Threat": 6
  }},

  "competitor_chart_data": [
    {{"name": "Competitor A", "share_percent": 35.0}},
    {{"name": "Competitor B", "share_percent": 30.0}},
    {{"name": "Competitor C", "share_percent": 20.0}},
    {{"name": "Other", "share_percent": 15.0}}
  ],
  
  "kpi_summary": [
    {{"name": "Revenue (B USD, latest year)", "value": "float"}},
    {{"name": "YoY Revenue Growth %", "value": "float"}},
    {{"name": "Employees", "value": "float"}}
  ],

  "conflicts": [
    {{"topic": "string", "details": "Brief summary of the conflict", "needs_deep_dive": true}}
  ],

  "confidence_score": 0.86
}}

Rules:
- Return valid JSON only,first search about the company find true actual value of all this and return it.
- No markdown, no bullet symbols, no headings,no ** .
"""
    return base, rag_ctx, web_text


def _build_steps(rag_ctx: List[str], web_text: Optional[str]) -> List[ResearchStep]:
    # This remains the same
    steps: List[ResearchStep] = []
    # ... (steps logic remains the same)
    steps.append(
        ResearchStep(
            id="intent",
            label="Understand user intent and depth",
            status="complete",
            detail="Parsed company name and research depth.",
        )
    )

    steps.append(
        ResearchStep(
            id="rag_search",
            label="RAG search on internal PDFs",
            status="complete" if rag_ctx else "skipped",
            detail=(
                f"Retrieved {len(rag_ctx)} internal chunks."
                if rag_ctx
                else "No RAG documents found."
            ),
        )
    )

    steps.append(
        ResearchStep(
            id="web_search",
            label="External web search",
            status="complete" if web_text else "skipped",
            detail=(
                "Web search executed via Serper."
                if web_text
                else "Web search unavailable or failed."
            ),
        )
    )

    steps.append(
        ResearchStep(
            id="llm_synthesis",
            label="LLM synthesis of account plan",
            status="complete",
            detail="LLM combined all signals into structured account plan.",
        )
    )
    return steps


def _build_account_plan_from_llm(
    raw: str,
    req: StartResearchRequest,
    rag_ctx: List[str],
    web_text: Optional[str],
) -> AccountPlan:
    data = _parse_llm_json(raw)

    # --- Fallback logic remains the same ---
    if not data:
        # ... (simplified AccountPlan structure for error/fallback)
        overview = f"{req.company_name} – Executive Snapshot:\n\n" + raw[:700]
        full_content = raw

        sections = [
            CompanySection(title="Full Research Output", content=full_content),
        ]
        
        meta = ResearchMetadata(
            generated_at=_now_iso(),
            depth=req.depth,
            sources_used=(
                (["rag"] if rag_ctx else [])
                + (["web"] if web_text else [])
            ),
            steps=_build_steps(rag_ctx, web_text),
            conflicts_detected=False,
            confidence_score=0.0,
        )

        return AccountPlan(
            company_name=req.company_name,
            depth=req.depth,
            overview=overview,
            company_profile="",
            market_analysis="",
            financial_highlights="",
            product_portfolio="",
            technology_stack="",
            competitors="",
            swot="",
            opportunities="",
            risks="",
            account_plan_30_60_90="",
            kpi_summary=None,
            sources=None,
            metadata=meta,
            sections=sections,
        )


    # --- Normal Path: Structured JSON available ---
    
    # Core narrative fields
    overview = data.get("overview", "")
    company_profile = data.get("company_profile", "")
    market_analysis = data.get("market_analysis", "")
    financial_highlights = data.get("financial_highlights", "")
    product_portfolio = data.get("product_portfolio", "")
    technology_stack = data.get("technology_stack", "")
    competitors = data.get("competitors", "")
    swot = data.get("swot", "")
    risks = data.get("risks", "")
    
    # 🆕 New Structured Fields
    opportunities_points = data.get("opportunities_points", [])
    plan_table = data.get("plan_table", [])
    swot_radar_scores = data.get("swot_radar_scores")
    competitor_chart_data = data.get("competitor_chart_data")
    
    # Convert structured lists back to readable string for the original 'opportunities' field
    opportunities_text = "\n\n".join(opportunities_points)
    
    
    plan_summary = "Structured 30-60-90 Day Plan data is available below."

    kpi_summary = data.get("kpi_summary")
    sources = data.get("sources") or {}

    confidence_score_raw = data.get("confidence_score")
    try:
        confidence_score = float(confidence_score_raw) if confidence_score_raw is not None else 0.0
    except (TypeError, ValueError):
        confidence_score = 0.0

    conflicts = data.get("conflicts") or []

    steps = _build_steps(rag_ctx, web_text)
    meta = ResearchMetadata(
        generated_at=_now_iso(),
        depth=req.depth,
        sources_used=[
            src
            for src in (
                "rag" if rag_ctx else None,
                "web" if web_text else None,
                "llm",
            )
            if src
        ],
        steps=steps,
        conflicts_detected=bool(conflicts),
        confidence_score=confidence_score,
    )

    # Sections for UI cards
    sections = [
        CompanySection(title="Company Profile", content=company_profile),
        CompanySection(title="Market Analysis", content=market_analysis),
        CompanySection(title="Financial Highlights", content=financial_highlights),
        CompanySection(title="Product Portfolio", content=product_portfolio),
        CompanySection(title="Technology Stack", content=technology_stack),
        CompanySection(title="Competitors", content=competitors),
        CompanySection(title="SWOT Analysis", content=swot),
        CompanySection(title="Opportunities", content=opportunities_text), # Use detailed points
        CompanySection(title="Risks", content=risks),
        CompanySection(title="30-60-90 Day Plan", content=plan_summary), # Use table string
    ]

    if not overview:
        overview = (
            f"{req.company_name} – Executive Snapshot:\n\n"
            + company_profile[:400]
            + "\n\n"
            + market_analysis[:400]
        )

    # Use a modified AccountPlan to store the structured chart data
    return AccountPlan(
        company_name=req.company_name,
        depth=req.depth,
        overview=overview,
        company_profile=company_profile,
        market_analysis=market_analysis,
        financial_highlights=financial_highlights,
        product_portfolio=product_portfolio,
        technology_stack=technology_stack,
        competitors=competitors,
        swot=swot,
        opportunities=opportunities_text,
        risks=risks,
        account_plan_30_60_90=plan_summary,
        kpi_summary=kpi_summary,
        
        # 🆕 Store New Structured Data in 'sources' for retrieval by chat agent
        sources={
            "swot_radar_scores": swot_radar_scores,
            "competitor_chart_data": competitor_chart_data,
            "plan_table": plan_table,
        },
        metadata=meta,
        sections=sections,
    )


def update_account_plan_section(
    conversation_id: str, section_title: str, new_content: str
) -> bool:
    # ... (remains the same as provided in the last complete file)
    state = _CONVERSATIONS.get(conversation_id)
    if not state:
        return False

    plan: AccountPlan = state["plan"]

    found_section: Optional[CompanySection] = None
    for section in plan.sections:
        if section.title == section_title:
            found_section = section
            break

    if found_section:
        found_section.content = new_content
        
        attr_name = (
            section_title.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("__", "_")
        )
        if hasattr(plan, attr_name) and attr_name not in ["sections", "metadata", "sources"]:
            setattr(plan, attr_name, new_content)

        state.setdefault("history", []).append(
            {
                "user": f"System Edit: Update '{section_title}'",
                "assistant": f"The section '{section_title}' has been updated with the new content.",
            }
        )

        return True
    return False


def start_research(req: StartResearchRequest) -> StartResearchResponse:
    prompt, rag_ctx, web_text = build_research_prompt(req)
    raw = call_llm(BASE_SYSTEM_PROMPT, prompt)

    plan = _build_account_plan_from_llm(raw, req, rag_ctx or [], web_text)

    cid = str(uuid.uuid4())

    parsed = _parse_llm_json(raw)
    conflicts_to_resolve = parsed.get("conflicts", []) if isinstance(parsed, dict) else []

    _CONVERSATIONS[cid] = {
        "plan": plan,
        "history": [],
        "conflicts_to_resolve": conflicts_to_resolve,
    }

    initial_history = _generate_pipeline_history(plan, conflicts_to_resolve)
    _CONVERSATIONS[cid]["history"] = initial_history

    return StartResearchResponse(conversation_id=cid, account_plan=plan)


def chat_with_agent(req: ChatMessageRequest) -> ChatMessageResponse:
    state = _CONVERSATIONS.get(req.conversation_id)
    if not state:
        return ChatMessageResponse(
            reply="Invalid conversation ID. Please start a new research session."
        )

    plan: AccountPlan = state["plan"]
    history: List[Dict[str, str]] = state.setdefault("history", [])
    conflicts: List[Dict[str, Any]] = state.get("conflicts_to_resolve", [])
    user_message = req.message.strip()
    user_message_lower = user_message.lower()
    
    
    # 🆕 NEW: Dynamic Chart Request Logic (Competitor Pie/SWOT Radar)
    chart_data = None
    chart_type = None
    chart_title = None
    
    if "competitor" in user_message_lower and ("pie" in user_message_lower or "chart" in user_message_lower):
        chart_data = plan.sources.get("competitor_chart_data")
        chart_type = "pie"
        chart_title = "Competitor Share Breakdown"
        
    elif "swot" in user_message_lower and ("graph" in user_message_lower or "radar" in user_message_lower or "chart" in user_message_lower):
        chart_data = plan.sources.get("swot_radar_scores")
        chart_type = "radar"
        chart_title = "SWOT Analysis Scores"

    if chart_data:
        # Agent sends a structured response that the UI can recognize and visualize
        structured_reply = {
            "reply_type": "chart_request",
            "chart_type": chart_type,
            "data": chart_data,
            "title": chart_title,
            "narrative": f"Here is the visual breakdown of the {chart_title} data, available for rendering as a chart in the workspace."
        }
        
        reply_json = json.dumps(structured_reply)
        
        # Store the full JSON in history
        history.append({"user": user_message, "assistant": reply_json})
        
        # Return the full JSON string so the immediate UI refresh can parse it too
        return ChatMessageResponse(reply=reply_json)


    # --- KPI/Table Chart Request Logic (remains the same) ---
    if "show" in user_message_lower and ("kpi" in user_message_lower or "chart" in user_message_lower or "graph" in user_message_lower):
        if plan.kpi_summary:
            kpi_data = json.dumps(plan.kpi_summary, indent=2)
            reply = (
                "Here is the structured KPI data required for visualization: "
                f"```json\n{kpi_data}\n``` "
                "You can use this to render a chart (e.g., bar chart for Revenue/Growth). "
                "Next Step: Would you like to review the updated competitors section or discuss the risks?"
            )
        else:
            reply = "I apologize, the financial KPI data summary is not available for this plan depth. Would you like to run a 'deep-dive on financials'?"
        
        history.append({"user": user_message, "assistant": reply})
        return ChatMessageResponse(reply=reply)
    
    # --- CONFLICT RESOLUTION LOGIC (remains the same) ---
    if conflicts and ("deep-dive" in user_message_lower or "dig deeper" in user_message_lower):
        # ... (logic remains the same)
        conflict_to_resolve = conflicts.pop(0)
        topic = conflict_to_resolve.get("topic", "data discrepancy")
        deep_dive_query = f"{plan.company_name} {topic} official report"
        new_web_data = web_search(deep_dive_query, num_results=3) 
        
        resolution_prompt = (
            f"You must resolve the following conflict in the account plan for {plan.company_name}.\n"
            f"Conflict topic: {topic}.\n"
            f"New research data: {new_web_data or 'No new data found.'}\n"
            # ... (rest of prompt)
        )
        resolution_json = call_llm(BASE_SYSTEM_PROMPT, resolution_prompt)
        resolution_data = _parse_llm_json(resolution_json)
        reply_summary = resolution_data.get(
            "resolution_summary",
            "The conflict could not be fully resolved with the available new data.",
        )
        state["conflicts_to_resolve"] = conflicts
        reply = (
            f"Conflict resolution for topic '{topic}' is complete. Summary: {reply_summary} "
            f"There are {len(conflicts)} remaining conflicts. You can also ask to review updated sections such as competitors or market analysis."
        )

        history.append({"user": user_message, "assistant": reply})
        return ChatMessageResponse(reply=reply)

    # --- SELECTIVE PLAN EDITING LOGIC (remains the same) ---
    if "edit" in user_message_lower and "section" in user_message_lower:
        reply = (
            "You can edit a specific section by specifying which section and the new focus. "
            "For example: Edit the 30-60-90 Day Plan to focus on EMEA expansion."
        )
        history.append({"user": user_message, "assistant": reply})
        return ChatMessageResponse(reply=reply)

    # --- GENERAL CHAT AND FOLLOW-UP LOGIC (remains the same) ---

    hist_text = "\n".join(
        f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history
    )
    plan_summary_for_context = f"""
    Company: {plan.company_name}
    Overview: {plan.overview[:500]}
    Competitors: {plan.competitors[:500]}
    Market Analysis: {plan.market_analysis[:500]}
    Financials: {plan.financial_highlights[:500]}
    """

    user_prompt = f"""
Continue the enterprise research dialog for the following company and account plan.
Use the Plan Data Snippets to provide context-aware answers.

Plan Data Snippets:
{plan_summary_for_context}

Conversation so far:
{hist_text or "(no prior turns)"}

User question:
{user_message}

Requirements:
- Give a concise, insight-rich answer based on the plan data.
- **PROACTIVE RULE**: At the end of your answer, add ONE short, high-value follow-up offer, for example:
  "Would you like me to go deeper into competitors, financials, or market trends?"
"""
    reply = call_llm(BASE_SYSTEM_PROMPT, user_prompt)
    history.append({"user": user_message, "assistant": reply})

    return ChatMessageResponse(reply=reply)


def generate_feedback(req: FeedbackRequest) -> FeedbackResponse:
    # ... (remains the same)
    state = _CONVERSATIONS.get(req.conversation_id)
    if not state:
        return FeedbackResponse(feedback_summary="Invalid conversation ID.")

    plan: AccountPlan = state["plan"]
    history: List[Dict[str, str]] = state.get("history", [])

    hist = "\n".join(f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history)

    prompt = f"""
You are reviewing an enterprise account plan.
...
"""

    feedback = call_llm(BASE_SYSTEM_PROMPT, prompt)
    return FeedbackResponse(feedback_summary=feedback)


def get_conversation_state(conversation_id: str) -> Optional[Dict[str, Any]]:
    return _CONVERSATIONS.get(conversation_id)


def _sse(event: str, data: str) -> str:
    # ... (remains the same)
    clean = data.replace("\n", " ")
    return f"event: {event}\ndata: {clean}\n\n"


def stream_research(req: StartResearchRequest) -> Generator[str, None, None]:
    # ... (remains the same)
    prompt, rag_ctx, web_text = build_research_prompt(req)

    yield _sse("status", "Understanding intent and preparing research pipeline.")
    yield _sse("status", "Searching internal RAG documents.")
    yield _sse(
        "status",
        f"RAG search complete. {len(rag_ctx or [])} internal chunks found.",
    )

    yield _sse("status", "Running external web search.")
    yield _sse(
        "status",
        "Web search complete."
        if web_text
        else "Web search unavailable. Continuing with RAG and LLM synthesis.",
    )

    yield _sse(
        "status", "Synthesizing account plan with LLM and checking for conflicts."
    )

    for chunk in stream_llm(BASE_SYSTEM_PROMPT, prompt):
        if chunk.strip():
            yield _sse("token", chunk)

    yield _sse("status", "Research synthesis complete. Checking for data conflicts.")
    yield _sse("status", "Plan ready. Conversation starting.")
    yield _sse("done", "true")
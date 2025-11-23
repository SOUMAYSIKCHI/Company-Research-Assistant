import os
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .models import (
    StartResearchRequest,
    StartResearchResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    FeedbackRequest,
    FeedbackResponse,
)

from .services.research_service import (
    start_research,
    chat_with_agent,
    generate_feedback,
    get_conversation_state,
    stream_research,
    update_account_plan_section, # <-- Crucial for editing
)

app = FastAPI(title="Company Research Assistant (Groq + RAG)")


# ------------------------------------------------------------
# CORS (allow frontend to connect)
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Templates & Static
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ------------------------------------------------------------
# JSON API ROUTES
# ------------------------------------------------------------

@app.post("/api/research/start", response_model=StartResearchResponse)
def start(req: StartResearchRequest):
    return start_research(req)


@app.post("/api/research/chat", response_model=ChatMessageResponse)
def chat(req: ChatMessageRequest):
    return chat_with_agent(req)


@app.post("/api/research/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    return generate_feedback(req)


@app.post("/api/research/stream")
def stream(req: StartResearchRequest):
    """
    Streaming research endpoint (Server-Sent Events).
    """
    generator = stream_research(req)
    return StreamingResponse(generator, media_type="text/event-stream")


# ------------------------------------------------------------
# UI ROUTES (HTML Frontend)
# ------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/about")
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/help")
def help(request: Request):
    return templates.TemplateResponse("help.html", {"request": request})



@app.get("/ui/research", response_class=HTMLResponse)
def research_input(request: Request, error: str | None = None):
    company = request.query_params.get("company", "")
    return templates.TemplateResponse(
        "research_input.html",
        {
            "request": request,
            "error": error,
            "company": company,
        },
    )


@app.post("/ui/research/start")
def research_start_ui(
    request: Request,
    company_name: str = Form(...),
    depth: str = Form(...),
    extra_instructions: str = Form(""),
):
    if not company_name.strip():
        url = request.url_for("research_input") + "?error=Company+name+is+required"
        return RedirectResponse(url=url, status_code=303)

    req = StartResearchRequest(
        company_name=company_name.strip(),
        depth=depth,  # validated by Pydantic
        extra_instructions=extra_instructions or None,
    )
    resp = start_research(req)

    return RedirectResponse(
        url=request.url_for("research_results", conversation_id=resp.conversation_id),
        status_code=303,
    )


@app.get("/ui/research/{conversation_id}", response_class=HTMLResponse)
def research_results(request: Request, conversation_id: str):
    state = get_conversation_state(conversation_id)
    if state is None:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "Invalid or expired conversation ID.",
            },
            status_code=404,
        )

    plan = state["plan"]
    history = state["history"]

    # 🔒 Backend safety: guarantee JSON-safe field for Jinja
    if "sources" in plan.__dict__:
        plan.sources.setdefault("swot_radar_scores", {})

    return templates.TemplateResponse(
        "research_results.html",
        {
            "request": request,
            "conversation_id": conversation_id,
            "plan": plan,
            "history": history,
        },
    )

    state = get_conversation_state(conversation_id)
    if state is None:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "Invalid or expired conversation ID.",
            },
            status_code=404,
        )

    plan = state["plan"]
    history = state["history"]

    return templates.TemplateResponse(
        "research_results.html",
        {
            "request": request,
            "conversation_id": conversation_id,
            "plan": plan,
            "history": history,
        },
    )


@app.post("/ui/research/{conversation_id}/chat")
def research_chat_ui(
    request: Request,
    conversation_id: str,
    user_message: str = Form(...),
):
    _ = chat_with_agent(
        ChatMessageRequest(conversation_id=conversation_id, message=user_message)
    )
    return RedirectResponse(
        url=request.url_for("research_results", conversation_id=conversation_id),
        status_code=303,
    )

@app.post("/ui/research/{conversation_id}/edit")
def research_edit_ui(
    request: Request,
    conversation_id: str,
    section_title: str = Form(...),
    new_content: str = Form(...),
):
    # This route handles the selective plan update
    update_account_plan_section(
        conversation_id=conversation_id,
        section_title=section_title.strip(),
        new_content=new_content.strip(),
    )
    
    # Redirect back to the results page to show the updated content and chat message
    return RedirectResponse(
        url=request.url_for("research_results", conversation_id=conversation_id),
        status_code=303,
    )


@app.post("/ui/research/{conversation_id}/feedback")
def generate_feedback_ui(
    request: Request,
    conversation_id: str,
    overall_notes: str = Form(""),
):
    _ = generate_feedback(
        FeedbackRequest(conversation_id=conversation_id, overall_notes=overall_notes)
    )
    return RedirectResponse(
        url=request.url_for("feedback_page", conversation_id=conversation_id),
        status_code=303,
    )


@app.get("/ui/feedback/{conversation_id}", response_class=HTMLResponse)
def feedback_page(request: Request, conversation_id: str):
    state = get_conversation_state(conversation_id)
    if state is None:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "Invalid or expired conversation ID.",
            },
            status_code=404,
        )

    feedback = generate_feedback(FeedbackRequest(conversation_id=conversation_id))

    return templates.TemplateResponse(
        "feedback.html",
        {
            "request": request,
            "conversation_id": conversation_id,
            "plan": state["plan"],
            "feedback_summary": feedback.feedback_summary,
        },
    )


@app.get("/ui/research/{conversation_id}/download")
def download_report(conversation_id: str):
    from fastapi.responses import PlainTextResponse

    state = get_conversation_state(conversation_id)
    if state is None:
        return PlainTextResponse("Invalid conversation id", status_code=404)

    plan = state["plan"]

    lines = [
        f"Company Research Report: {plan.company_name}",
        "",
        plan.overview,
        "",
    ]
    for section in plan.sections:
        lines.append(f"## {section.title}")
        lines.append(section.content)
        lines.append("")

    text = "\n".join(lines)
    headers = {
        "Content-Disposition": f'attachment; filename="{plan.company_name}_report.txt"'
    }
    return PlainTextResponse(text, headers=headers)
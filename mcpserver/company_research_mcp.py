from mcp.server.fastmcp import FastMCP

from backend.models import ResearchDepth, StartResearchRequest, ChatMessageRequest, FeedbackRequest
from backend.services.research_service import (
    start_research,
    chat_with_agent,
    generate_feedback,
)

mcp = FastMCP("company-research")


@mcp.tool()
async def start_company_research(
    company_name: str,
    depth: ResearchDepth,
    extra_instructions: str | None = None,
):
    req = StartResearchRequest(
        company_name=company_name,
        depth=depth,
        extra_instructions=extra_instructions,
    )
    return start_research(req).model_dump()


@mcp.tool()
async def chat_with_research_agent(conversation_id: str, message: str):
    req = ChatMessageRequest(conversation_id=conversation_id, message=message)
    return chat_with_agent(req).model_dump()


@mcp.tool()
async def generate_research_feedback(conversation_id: str, notes: str | None = None):
    req = FeedbackRequest(conversation_id=conversation_id, overall_notes=notes)
    return generate_feedback(req).model_dump()


def main():
    mcp.run()


if __name__ == "__main__":
    main()

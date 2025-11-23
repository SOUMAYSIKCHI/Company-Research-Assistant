# 📘 **README.md — Company Research Assistant**

### *Enterprise-Grade Company Research & Account Plan Generation (Groq + LangChain + MCP + RAG + Web Scraping)*

---

## 🚀 **Overview**

**Company(AI) Research Assistant** is an enterprise-focused, agentic system that conducts deep company research, synthesizes multi-source intelligence, detects conflicting information, and generates structured account plans.
It is built for sales, strategy, GTM, and consulting workflows requiring **fast, accurate, and explainable** research.

The platform combines:

* **LLM intelligence (Groq Llama 3.1)**
* **RAG (LangChain + Chroma)**
* **MCP Tools** for agentic behaviour
* **Serper Web Search** for external data
* **FastAPI + Jinja2** UI
* **Interactive Chat**, **Conflict Resolution**, **Section Editing**, and **Chart Generation**

This application is designed with **high conversational quality**, **proactive agent behaviour**, and **adaptability**, directly aligned with the hackathon evaluation criteria.

---

# 📄 **Documentation**

## **PRD — Product Requirements Document** :

https://drive.google.com/file/d/17LZm0odnt4ouAGqWfoYKSM-Mi5wVHO9V/view?usp=sharing

## **Technical documentation** :

https://drive.google.com/file/d/12YYwqG1PMnbMLTkldrKt9nryThJqRIP0/view?usp=sharing

---

## 🧠 **Core Capabilities**

### **1. Multi-Source Research Pipeline**

The AI agent gathers research from:

* **Internal RAG documents (PDFs)** via `Chroma` vector database
* **External Web Search (Serper)**
* **Structured LLM synthesis** using Groq’s high-speed inference
* **Custom MCP Tools** enabling multi-step agentic workflows 

### **2. Conflict Detection & Deep-Dive Resolution**

The LLM is instructed to **detect conflicting financials, metrics, or strategic statements** and surface them as structured JSON:

* `"conflicts": [{ topic, details, needs_deep_dive }]`

Users can command the AI to *“deep dive on revenue numbers”* and the agent automatically fetches new data and resolves inconsistencies.

### **3. Structured Account Plan Generation**

The research output includes:

* Executive overview
* Market analysis
* Company profile
* Financial highlights
* Competitors
* SWOT
* Opportunities (detailed)
* Risks
* 30-60-90 day GTM plan
* KPI summaries
* Radar & pie-chart-ready data
* Source metadata and research steps 

### **4. Conversational Workflow (Chat UI)**

* The agent **remembers previous turns**
* Provides **insight-rich responses**
* Follows up proactively:
  *“Would you like me to go deeper into competitors, financials, or market trends?”*

### **5. Selective Section Editing**

Users can modify any section of the generated plan via UI or chat.
The system updates both the **AccountPlan model** and **UI card content**.


### **6. Chart Data Generation**

The agent can return structured data for:

* **SWOT Radar Charts**
* **Competitor Pie Charts**
* **KPI Bar Charts**

These are stored in `plan.sources` and accessible through chat.


### **7. Full UI (FastAPI + Jinja2)**

Frontend routes include:

* `/ui/research` – enter company & depth
* `/ui/research/:id` – view analysis
* `/ui/research/:id/chat` – chat with the agent
* `/ui/research/:id/edit` – update sections
* `/ui/feedback/:id` – plan evaluation


---

## 🏛️ **Architecture Overview**

### **System Diagram**

```
                       ┌──────────────────────────┐
                       │         Frontend          │
                       │  FastAPI + Jinja2 HTML    │
                       └──────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │     FastAPI Backend    │
                         │ (REST, SSE, UI routes) │
                         └─────────────┬──────────┘
                                       │
   ┌───────────────────────────────────┼─────────────────────────────┐
   │                                   │                             │
   ▼                                   ▼                             ▼
┌───────────┐                 ┌────────────────┐            ┌───────────────────┐
│ RAG Layer │                 │  MCP Tools     │            │  LLM Engine (Groq)│
│ LangChain │                 │ start/chat/... │            │ llama-3.1-8b       │
│ Chroma DB │                 │ agentic actions│            │ streaming + sync   │
└───────────┘                 └────────────────┘            └───────────────────┘
      │                                 │                             │
      └────────────────────────┬────────┴───────────────┬────────────┘
                               ▼                        ▼
                     ┌────────────────┐       ┌─────────────────────┐
                     │ Web Search     │       │ Structured JSON Plan│
                     │ Serper API     │       │ + Conflicts + Charts│
                     └────────────────┘       └─────────────────────┘
```

---

## ⚙️ **Tech Stack**

### **Backend**

* **FastAPI**
* **Groq LLM** (Llama-3.1-8B Instant) 
* **LangChain** (HuggingFace Embeddings + Chroma vector DB) 
* **MCP (Model Context Protocol)** custom tools for research, chat, and feedback
* **Serper API Web Search**

### **Frontend**

* **Jinja2 templates**
* **HTML/CSS/JS**
* **Chart Rendering (SWOT Radar, Competitor Pie, KPI Bar)**

### **Data**

* In-memory conversation store (`_CONVERSATIONS`)
* Persisted Chroma DB for RAG

---

## 🔄 **End-to-End Flow**

### **1. User Inputs Company Name**

Route: `/ui/research/start`
Backend constructs a **consolidated research prompt** using:

* RAG chunks
* Web summary
* Custom JSON schema
* Conflict detection instructions


### **2. LLM Executes Multi-Source Synthesis**

LLM produces:

* Narrative fields
* SWOT
* Competitors
* KPIs
* 30-60-90 plan
* Radar/pie data
* Conflicts array
* Confidence score

### **3. AccountPlan Object Created**

`AccountPlan` is built from structured JSON.


### **4. Pipeline History & Chat Initialized**

System auto-generates pipeline steps for transparency.


### **5. User Interacts with Agent**

They can:

* Ask questions
* Request charts
* Deep-dive conflicts
* Edit sections

### **6. Feedback + Download**

User can export plan or get AI-generated review.

---

## 🧪 User Interaction Modes : 

### ✔ Confused User

Agent clarifies scope, intent, and provides examples.

### ✔ Efficient User

Agent generates summaries quickly and offers shortcuts.

### ✔ Chatty User

Agent gently redirects to goal while preserving context.

### ✔ Edge Case User

Invalid inputs → safe fallback with guided suggestions.

---

## 🚀 **Running the Project Locally**

### **1. Install Dependencies**

```
pip install -r requirements.txt
```

### **2. Environment Variables**

```
GROQ_API_KEY=
SERPER_API_KEY=
SERPER_URL=https://google.serper.dev/search
```

### **3. Start Backend**

```
uvicorn backend.app:app --reload
```

### **4. Access UI**

```
http://localhost:8000/ui/research
```

---

## 📁 **Project Structure**

```
backend/
 ├── app.py                     # FastAPI routes + UI flow
 ├── models.py                  # Pydantic data models
 ├── services/
 │     ├── llm_service.py       # Groq LLM integration
 │     ├── rag_service.py       # RAG pipeline (LangChain + Chroma)
 │     └── research_service.py  # Full research + synthesis pipeline
 ├── mcp_server/
 │     └── company_research_mcp.py   # MCP tools for research, chat & feedback
 ├── templates/                 # Frontend templates (Jinja2)
 └── static/                    # CSS, JS, assets

```

---

## 📜 Design Decisions : 

### **1. Conversational Quality First**

* All responses are context-aware
* Follow-up prompts embedded in model logic
* No hallucinated charts — only derived chart data
* Conflict awareness improves reliability

### **2. Agentic Behaviour**

* Custom MCP tools let the agent act
* Step logging shows transparency
* Conflict resolution loop supports autonomy

### **3. Technical Implementation**

* High-performance Groq inference
* Modern LangChain RAG stack
* Clean modular architecture
* SSE streaming support for real-time research

### **4. Adaptability**

* Handles confused users
* Error-tolerant JSON parsing
* Fallback modes for empty RAG/Web results

---

# 🎥 Demo Video
A short end-to-end demonstration of the **Company (AI) Research Assistant**, showcasing the agentic research workflow, multi-source synthesis, UI interactions, and feedback generation.

🔗 **Demo Link:** _Coming Soon_ / _Add Your Link Here_

---

# 📸 UI Preview & Application Screenshots

Below are real UI screenshots from the **Company (AI) Research Assistant** showcasing  
the research workflow, agent interactions, results dashboard, feedback pages, and edit flow.

All images are located in:  
`/assets/screenshots/`

---

## 🖥️ **Dashboard & Research Flow** : 

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222527.png" width="85%" />
</p>

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222540.png" width="85%" />
</p>

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222612.png" width="85%" />
</p>

---

## 🔍 **Research Results – Account Plan View**

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222621.png" width="85%" />
</p>

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222630.png" width="85%" />
</p>

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222703.png" width="85%" />
</p>

---

## 💬 **Chat Interaction & Agentic Updates**

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222709.png" width="85%" />
</p>

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222755.png" width="85%" />
</p>

---

## 📊 **Feedback, Metrics & Charts**

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222808.png" width="85%" />
</p>

---

## ✏️ **Section Editing & Updates**

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222817.png" width="85%" />
</p>

<p align="center">
  <img src="assets/Screenshot 2025-11-23 222829.png" width="85%" />
</p>


---

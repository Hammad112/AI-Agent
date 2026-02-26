# 🤖 Generic AI Agent for Orders & Scheduling

A high-performance, generic Python backend framework for AI agents capable of handling **Orders**, **Scheduling**, and **Q&A** for any small business. 

Simply input one PDF describing the business (menu, services, policies), and the agent automatically configures itself to handle real customer interactions.

---

## 🏗️ Architecture Overview

The agent is built using **LangGraph** as a directed state machine, ensuring complex multi-intent messages are handled with standard business logic.

### Multi-Node Agentic Pipeline

```mermaid
graph TD
    Input[User Query] --> Planner[Planner Node]
    Planner --> Intent[Intent Detector]
    Intent --> RAG[Non-Embedding RAG]
    RAG --> Router[Custom Batched Router]
    Router --> Tools[Execute Tools]
    Tools --> Generator[Response Generator]
    Generator --> Critic[Critic Audit Node]
    Critic -- Fail --> Generator
    Critic -- Pass --> End[Final Response]
```

| Node                | Description                                                         |
| :------------------ | :------------------------------------------------------------------ |
| **Planner**         | Generates an internal strategy before acting.                       |
| **Intent Detector** | Identifies multiple intents (e.g., "Add a pizza AND book a table"). |
| **RAG Retrieval**   | LLM-based retrieval without vector embeddings/indexes.              |
| **Custom Router**   | Batched tool detection (not LangChain's default).                   |
| **Critic**          | Validates the response against PDF rules before sending.            |

---

## ✨ Key Features

- **🚀 Generic Implementation**: Works for any business (Pizza shop, Dental Clinic, Dry Cleaner).
- **🧠 Knowledge Enrichment**: Automatically generates supplementary FAQ and "skills" articles that aren't in the original PDF.
- **📊 Synthetic CRM**: Simulates realistic customer data, order history, and family members for testing purposes.
- **📅 Intelligent Scheduling**: Parses natural dates ("next Friday at 2pm") and checks provider availability.
- **🛒 Cart Management**: Full order lifecycles including modifiers (e.g., "extra cheese"), address validation, and loyalty points.
- **🔍 Non-Embedding RAG**: Uses LLM logic to match query topics to document paragraphs for clinical accuracy.

---

## 🛠️ Tech Stack

- **Framework**: LangGraph, LangChain
- **LLM**: OpenAI GPT-4o-mini (Primary), Google Gemini 2.0 (Fallback)
- **Database**: SQLite (11 tables)
- **Utilities**: PyMuPDF (PDF parsing), Faker (Synthetic data), Geopy (Address validation)

---

## 💻 Windows Setup Guide

### 1. Prerequisites
- **Python 3.11+**
- **Git** (optional)
- An **OpenAI** or **Gemini** API Key

### 2. Installation
Open PowerShell in the project directory:

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY or OPENAI_API_KEY
```

### 3. Running the Agent (CLI)
```powershell
# Start the interactive assistant
python main.py

# Or load a specific business description
python main.py --pdf sample_pdfs/mario_pizza.md
```

### 4. Running the API Server
```powershell
python app/server.py
# Access documentation at http://localhost:8000/docs
```

---

## 📂 Project Structure

```bash
agent/
├── main.py                     # Main CLI entry point
├── agent/
│   ├── agent.py                # LangGraph definition (7 nodes)
│   ├── tools.py                # 26 modular agent tools
│   └── rag_engine.py           # LLM-only retrieval logic
├── core/
│   ├── database.py             # SQLite schema & Synthetic data
│   ├── llm_client.py           # Unified model wrapper
│   └── logger.py               # Detailed calculation logging
├── processing/
│   ├── pdf_processor.py        # Structural chunking logic
│   └── knowledge_enricher.py   # LLM "Skills" generator
└── sample_pdfs/                # Pre-loaded business examples
```

---

## 📝 Testing & Validation

Run the automated smoke test to verify the system's performance on multiple business types:
```powershell
python tests/smoke_test.py
```
This script validates:
1. PDF Ingestion & Enrichment.
2. Synthetic User Creation.
3. Multi-turn Chat Logic.

---
*Developed for Hammad Nasir - Generic AI Agent Project*

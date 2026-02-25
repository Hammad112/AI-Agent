# Generic AI Agent for Orders & Scheduling

A modular, RAG-based AI assistant built with LangChain and LangGraph. It runs entirely on business rules extracted from a PDF and supports multi-turn ordering, scheduling, and Q&A.

## Features
- **Generic RAG**: No embeddings used. Employs LLM-based topic matching and contextual retrieval.
- **Agentic Pipeline**: Includes Planner, Smart Router, and Critic/Evaluator nodes.
- **Structural Logging**: Complete audit trail of LLM calls, tool usage, and knowledge retrieval in SQLite.
- **Smart Tools**: Advanced handling of orders, appointments, loyalty, and disputes.
- **Address Validation**: Validates Canadian addresses and postal codes using free geolocation services.

## Installation (Windows)

1. **Clone the repository** and navigate to the project folder.
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment**:
   - Copy `.env.example` to `.env`.
   - Add your `GEMINI_API_KEY` or `OPENAI_API_KEY`.
   - (Optional) Set `DB_NAME`.

## Usage

### Starting the Agent
Run the main entry point and provide a business PDF:
```bash
python main.py --pdf sample_pdfs/mario_pizza.md
```
Or run interactively:
```bash
python main.py
```

### Running Server
To start the FastAPI backend:
```bash
uvicorn app.main:app --port 5000 --reload
```

## Testing
Run the comprehensive smoke test suite:
```bash
python tests/smoke_test.py
```

## Project Structure
- `main.py`: Interactive CLI entry point.
- `agent/`: LangGraph state machine, tools, and RAG engine.
- `core/`: Database logic, LLM client, and logging.
- `db/`: Database schema and migration scripts.
- `processing/`: PDF parsing and knowledge enrichment (skills generation).
- `tests/`: End-to-end smoke tests and verification scripts.

## Deliverables
- `Project_Documentation.docx`: Detailed installation and usage guide.
- `business_agent.db`: SQLite database with persistent state.
- `agent_logs`: Full audit logs for every LLM interaction.

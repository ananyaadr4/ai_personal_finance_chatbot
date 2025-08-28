# ARCHITECTURE

## Overview
This app has two components:
- **FastAPI backend** (`main.py`) — parses CSVs, computes metrics, classifies categories (keyword-based), and answers questions via **Ollama** (default `llama3.2`). It keeps per‑upload data in memory (session‑scoped).
- **Static frontend** (`test-frontend.html`) — uploads CSVs and renders charts with **Chart.js**. It calls the API and displays chat responses.

## Data Flow (upload → metrics → chatbot → response)
1. **Upload CSV** → `POST /upload-transactions`
   - Validates columns (`date, description, amount, [category]`).
   - Normalizes names and parses dates; missing `category` is auto‑classified by keywords (Food/Transport/…).
   - Stores transactions under a generated `session_id`.
   - Returns stats (totals, category & monthly breakdowns, top merchants) + budget alerts.
2. **Metrics & Charts**
   - Frontend renders Category/Monthly/Top charts from the returned stats.
3. **Chat**
   - UI sends `POST /chat` with `{message, session_id}`.
   - Backend builds a financial **context** (totals, category %s, monthly aggregates, top merchants, biggest expense, date range).
   - It generates a **specialized prompt** and calls Ollama.
   - If AI is unavailable, a **fallback** rule‑based response is produced.
   - Optional `chart_data` (pie/line/bar) is returned to guide the UI.
4. **Response**
   - Frontend displays the AI text and toggles to the most relevant chart.

## Design Choices
- **Keyword classifier** for categories: transparent, fast, no model dependency.
- **Absolute spend** for analysis (`abs(amount)`): treats negative debits uniformly.
- **In‑memory session store**: simplest way to isolate users for a demo.
- **ChartData generator** server‑side: makes it easy for other clients to reuse the API.
- **CORS pinned to `http://localhost:3000`**: aligns with the dev server where the static HTML is hosted.

## Trade‑offs
- **Persistence**: no DB — data disappears on server restart; simplifies setup.
- **Classification accuracy**: keywords can mislabel edge cases; acceptable for demo.
- **Security**: no auth/throttling — okay for local demo, not production ready.
- **Scalability**: single‑process, in‑memory — suitable only for small datasets.
- **CORS rigidity**: avoids `*` in samples; requires hosting UI on port 3000.

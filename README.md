# FinanceBot — Personal Finance Chatbot (Frontend + Backend)

A small, self-contained web app that accepts a CSV of transactions, summarizes spending, and lets you ask natural‑language questions about your finances. The backend is **FastAPI** and the UI is a single static HTML file using **Chart.js**. For local AI reasoning, the API integrates with **Ollama** (default model: `llama3.2`).

> Files provided in this bundle: `main.py` (API), `test-frontend.html` (UI), `requirements.txt` (deps).

---

## 1) Quickstart

### Prereqs
- **Python 3.10 or 3.11 **

- **Ollama** running locally (https://ollama.com/). Pull a model once:
  ```bash
  ollama pull llama3.2
  ollama serve
  ```

### Create & activate a venv
**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


### Run the backend (FastAPI)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run the frontend
Serve `test-frontend.html` from **http://localhost:3000** so CORS matches the backend config.

**Option A (Python):**
```bash
# from the folder containing test-frontend.html
python -m http.server 3000
# open http://localhost:3000/test-frontend.html
```


## 2) Using the app

1. Open the UI at `http://localhost:3000/test-frontend.html`.
2. Upload a CSV with columns: `date, description, amount, [category]`.
3. Ask questions like:
   - “How much did I spend on food?”
   - “Show my monthly trend.”
   - “Biggest expenses?”

## 6) Project Structure

```
.
├── main.py                 # FastAPI backend
├── test-frontend.html      # Single-page UI (Chart.js)
├── requirements.txt        # Python dependencies
├── sample-transactions.csv # Example data
└── README.md               # This file
```

Happy building!

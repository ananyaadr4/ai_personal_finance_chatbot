# AI_USAGE

## Tools Used
- **ChatGPT** — design help, code review, project documentation.  
- **Claude** — helped in fixing **syntax errors**, building **FastAPI endpoints**, and assisting with **frontend design** ideas.  
- **Ollama (local LLM)** — runtime analysis in `/chat` endpoint (default model: `llama3.2`).  

---

## Prompts for Coding Assistance
- *ChatGPT*:  
  - “Design a FastAPI endpoint that accepts CSV upload and normalizes headers.”  
  - “Sketch a keyword-based transaction classifier for common consumer categories.”  
  - “Generate JS to render Category / Monthly / Top charts from stats with Chart.js.”  

- *Claude*:  
  - “Fix this syntax error in my FastAPI code.”  
  - “Help me structure FastAPI routes correctly.”  
  - “Suggest a simple frontend layout with file upload + chatbot box.”  

---

## AI in the Code
- The backend builds a **structured financial prompt** with stats → AI (Ollama) turns it into concise advice.  
- Claude was especially useful for **syntax correction** and making sure **FastAPI routes were valid**.  
- ChatGPT contributed to higher-level design and **refactoring for readability**.  
- Frontend suggestions (layout, event flow, Chart.js integration) came from **Claude + ChatGPT combined**.  

---

## Validation / Modification
- Ran the code locally after each AI suggestion to confirm it compiled and executed correctly.  
- Adjusted AI-generated code when CORS rules, NumPy versions, or CSV parsing didn’t match local reality.  
- Simplified frontend HTML/JS to keep dependencies minimal.  

---

## Where AI Helped
- Debugging syntax errors quickly (Claude).  
- Structuring FastAPI endpoints and routes.  
- Designing a simple and functional frontend UI.  
- Turning structured stats into **human-friendly summaries** with Ollama.  

## Where AI Didn’t Help
- Strict CSV parsing & validation logic (had to be deterministic).  
- Budget alert thresholds (hard-coded rules).  
- Ensuring CORS worked properly (manual trial & error).  

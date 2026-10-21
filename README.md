# MedQuery v2

Question answering over public health documents, with cited answers, abstention when evidence is weak, and claim-level hallucination checking.

> **Not medical advice.** MedQuery answers only from its source documents and does not diagnose conditions or give dosing instructions. Always consult a qualified health professional.

## Setup (PowerShell)

```powershell
uv sync
Copy-Item .env.example .env   # then add your ANTHROPIC_API_KEY
uv run pytest
```

Status: work in progress. Results will be added here only once a re-runnable script produces them.

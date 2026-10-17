# 🩸 MedLens

Upload a lab report → get a plain-language explanation of what's out of range and **which specialist to see**.
MedLens is a *navigator*, not a diagnostician.

Built for **AMD Developer Hackathon: ACT III** — track: *Vision & Multimodal AI*.

## Why it's interesting
Two layers, so the AI never invents a number:
1. **Deterministic layer** — parses values from the report (PDF / image / text) and flags them against a fixed reference table. Pure code, zero hallucination.
2. **Reasoning layer** — an **open LLM served on AMD** turns those structured flags into plain language, picks the specialist, and explains urgency.
3. **Safety floor** — urgency can only be *raised* by the deterministic layer, never lowered by the model; a "not a diagnosis" disclaimer is always shown.

## AMD workload (eligibility)
Inference runs on an **open model served on AMD infrastructure**, via an OpenAI-compatible endpoint:
- **vLLM on AMD Developer Cloud (ROCm)**, or
- **Fireworks AI** serving an open model.

The endpoint is set entirely through environment variables — the same code runs against either.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env            # then edit it (see below)
set -a && source .env && set +a
python app.py                   # opens on http://localhost:7860
```

## Run with Docker
```bash
docker build -t medlens .
docker run -p 7860:7860 \
  -e OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e MODEL="$MODEL" \
  medlens
```

## Configure the model endpoint
Edit `.env` (see `.env.example`):

**A) vLLM on AMD Developer Cloud**
```
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
MODEL=meta-llama/Llama-3.1-8B-Instruct
```
Start the server on the AMD GPU first:
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**B) Fireworks AI**
```
OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
OPENAI_API_KEY=fw_your_key
MODEL=accounts/fireworks/models/llama-v3p1-8b-instruct
```

## Demo
Use the ready-made reports in `samples/`:
- `normal_report.txt` → green, "routine"
- `abnormal_report.txt` → red, "urgent" (high glucose / HbA1c → Endocrinologist, etc.)

Paste the text or upload it, hit **Analyze**. If the model endpoint is down, the app still shows the deterministic analysis so the demo never dies.

## ⚠️ Disclaimer
Prototype for a hackathon. Not a medical device, not a diagnosis. Reference ranges are approximate and lab-/sex-dependent. Always consult a licensed professional.

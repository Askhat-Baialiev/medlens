# 🩸 MedLens

Photograph a lab report **in any language** → understand it **in your own language** (24 languages), see **how urgent** it is, **which specialist** to see, and **what to ask** the doctor.

Built for people who live far from home: e.g. a migrant in Rome who gets an Italian lab report and reads it in Russian, Urdu, Chinese or Pashto.
MedLens is a *navigator*, not a diagnostician.

Built for **AMD Developer Hackathon: ACT III** — track: *Vision & Multimodal AI*.

## Why it's interesting
Two layers, so the AI never invents a number:
1. **Deterministic layer** — parses values from the report (PDF / image / text) and flags them against a fixed reference table. Pure code, zero hallucination.
2. **Reasoning layer** — an **open LLM served on AMD** turns those structured flags into plain language, picks the specialist, and explains urgency.
3. **Safety floor** — urgency can only be *raised* by the deterministic layer, never lowered by the model; a "not a diagnosis" disclaimer is always shown.

## AMD workload (eligibility)
All AI runs on **AMD GPUs via AMD Token Factory** (OpenAI-compatible endpoint):
- **Vision model** `DeepSeek-V4.1-Flash` reads photos and scanned PDFs of lab reports.
- **Reasoning model** `DeepSeek-V4-Flash` turns the flagged values into plain language, a specialist, and an urgency level.

## Run locally
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_BASE_URL=https://developer.amd.com.cn/radeon/api/v1
export OPENAI_API_KEY=rc-...            # your AMD Token Factory team key
export MODEL=DeepSeek-V4-Flash
export VISION_MODEL=DeepSeek-V4.1-Flash
python app.py                           # http://localhost:7860
```

## Run with Docker
```bash
docker build -t medlens .
docker run -p 7860:7860 \
  -e OPENAI_BASE_URL -e OPENAI_API_KEY -e MODEL -e VISION_MODEL \
  medlens
```

## Demo
Use the ready-made reports in `samples/`:
- `normal_report.txt` → green, "routine"
- `abnormal_report.txt` → red, "urgent" (high glucose / HbA1c → Endocrinologist, etc.)
- `abnormal_report.png` → a photo of a lab printout, read by the vision model on AMD
- `italian_report.png` → an Italian lab report photo; pick any answer language (try Русский, اردو, 中文, پښتو)

**Answer languages (24):** English, Italian, Russian, Kyrgyz, Chinese (Simplified), Hindi, Pashto, Urdu, Arabic, Persian, Spanish, French, German, Portuguese, Ukrainian, Romanian, Albanian, Bengali, Punjabi, Tagalog, Turkish, Uzbek, Kazakh, Tajik. Right-to-left scripts are rendered RTL.

Paste the text or upload it, hit **Analyze**. If the model endpoint is down, the app still shows the deterministic analysis so the demo never dies.

## ⚠️ Disclaimer
Prototype for a hackathon. **Demo uses synthetic data only** — never upload real medical records to the shared endpoint. Not a medical device, not a diagnosis. Reference ranges are approximate and lab-/sex-dependent. Always consult a licensed professional.

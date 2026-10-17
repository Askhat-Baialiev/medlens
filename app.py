"""
MedLens — upload a lab report, get a plain-language explanation of what's off
and which specialist to see.

Architecture (what to tell the judges):
  1. Deterministic layer  — parses numeric values from the report and flags
     out-of-range / critical results against a fixed reference table. No LLM,
     so it can't hallucinate a number.
  2. Reasoning layer      — an open LLM served on AMD (vLLM on AMD Developer
     Cloud, or an open model via Fireworks AI) turns those structured flags
     into plain language, picks the right specialist, and sets urgency.
  3. Safety floor         — urgency can only be raised by the deterministic
     layer, never lowered by the model. A fixed "not a diagnosis" disclaimer
     is always shown.

The model endpoint is OpenAI-compatible and configured entirely via env vars,
so the exact same code runs against vLLM or Fireworks.
"""

import json
import os
import re

import gradio as gr
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Config — works with vLLM on AMD Developer Cloud OR Fireworks, no code change #
# --------------------------------------------------------------------------- #
# vLLM example:      OPENAI_BASE_URL=http://localhost:8000/v1   OPENAI_API_KEY=EMPTY
#                    MODEL=meta-llama/Llama-3.1-8B-Instruct
# Fireworks example: OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
#                    OPENAI_API_KEY=fw_...   MODEL=accounts/fireworks/models/llama-v3p1-8b-instruct
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

with open("reference_ranges.json", "r", encoding="utf-8") as f:
    REFERENCE = json.load(f)["markers"]

URGENCY_ORDER = {"routine": 0, "see_doctor_soon": 1, "urgent": 2}
URGENCY_LABEL = {
    "routine": "🟢 Routine — discuss at your next regular check-up",
    "see_doctor_soon": "🟠 See a doctor soon — some results are outside the normal range",
    "urgent": "🔴 Urgent — one or more results are far outside the normal range; seek medical care promptly",
}
DISCLAIMER = (
    "⚠️ **This is not a diagnosis.** MedLens explains lab numbers in plain "
    "language and points you to the right kind of doctor. It cannot replace a "
    "medical professional. If you feel unwell or are worried, contact a doctor "
    "or emergency services."
)


# --------------------------------------------------------------------------- #
# Layer 1 — deterministic extraction & flagging                               #
# --------------------------------------------------------------------------- #
def parse_values(text):
    """Find known markers in free text and flag them against reference ranges."""
    findings = []
    for line in text.splitlines():
        # split "Marker: value unit" so marker-name digits (e.g. HbA1c) aren't read as values
        label, sep, value_part = line.partition(":")
        if not sep:
            label, value_part = line, line
        low_label = label.lower()
        for key, spec in REFERENCE.items():
            # word-boundary match so "hb" doesn't fire inside "hba1c", "ast" inside "fasting", etc.
            if any(re.search(r"\b" + re.escape(alias) + r"s?\b", low_label) for alias in spec["aliases"]):
                search_space = value_part if sep else re.sub(
                    r"\b" + re.escape(spec["aliases"][0]) + r"\b", " ", value_part, flags=re.I)
                nums = re.findall(r"[-+]?\d*\.?\d+", search_space)
                if not nums:
                    continue
                value = float(nums[0])
                lo, hi = spec["low"], spec["high"]
                if value < lo:
                    status = "low"
                elif value > hi:
                    status = "high"
                else:
                    status = "normal"
                # crude "critical" heuristic for the urgency floor
                critical = value < lo * 0.6 or value > hi * 1.6
                findings.append({
                    "marker": key,
                    "value": value,
                    "unit": spec["unit"],
                    "reference": f"{lo}–{hi}",
                    "status": status,
                    "critical": critical,
                    "specialist": spec["specialist"],
                })
                break  # one marker per line
    return findings


def deterministic_urgency(findings):
    if any(f["critical"] for f in findings):
        return "urgent"
    if any(f["status"] != "normal" for f in findings):
        return "see_doctor_soon"
    return "routine"


# --------------------------------------------------------------------------- #
# Layer 2 — LLM reasoning on AMD                                              #
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are MedLens, a careful assistant that EXPLAINS blood/lab results \
in plain language and recommends which medical specialist a person should see. \
You never diagnose and you always err toward advising the person to see a doctor.

You are given lab findings that were already extracted and range-checked by a \
deterministic parser. Trust those numbers and statuses — do NOT invent or change them.

Return ONLY valid JSON, no markdown, no commentary, in exactly this shape:
{
  "overall_summary": "2-3 sentences, plain language, for a non-medical person",
  "findings": [
    {"marker": "<human name>", "value": "<value> <unit>", "status": "low|normal|high",
     "plain_explanation": "one sentence a layperson understands"}
  ],
  "recommended_specialist": "<single most relevant specialty>",
  "urgency": "routine|see_doctor_soon|urgent",
  "urgency_reason": "one short sentence"
}
Keep it calm and non-alarming, but be honest about out-of-range values."""


def call_model(findings):
    payload = {
        "deterministic_findings": findings,
        "deterministic_urgency": deterministic_urgency(findings),
    }
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# Input extraction (PDF / image / pasted text)                                #
# --------------------------------------------------------------------------- #
def extract_text(file_path, pasted_text):
    if pasted_text and pasted_text.strip():
        return pasted_text
    if not file_path:
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    if ext in (".png", ".jpg", ".jpeg"):
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(file_path))
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def render(result, det_findings):
    final_urgency = max(
        result.get("urgency", "routine"),
        deterministic_urgency(det_findings),
        key=lambda u: URGENCY_ORDER.get(u, 0),
    )
    lines = [f"## {URGENCY_LABEL[final_urgency]}", ""]
    if result.get("urgency_reason"):
        lines.append(f"*{result['urgency_reason']}*\n")
    lines.append(f"**Summary.** {result.get('overall_summary', '')}\n")
    lines.append(f"**👉 Recommended specialist: {result.get('recommended_specialist', 'General Practitioner')}**\n")
    lines.append("### Your results")
    lines.append("| Marker | Value | Status | What it means |")
    lines.append("|---|---|---|---|")
    icon = {"low": "🔽 low", "high": "🔼 high", "normal": "✅ normal"}
    for f in result.get("findings", []):
        lines.append(
            f"| {f.get('marker','')} | {f.get('value','')} | "
            f"{icon.get(f.get('status','normal'),'')} | {f.get('plain_explanation','')} |"
        )
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def analyze(file_path, pasted_text):
    text = extract_text(file_path, pasted_text)
    if not text.strip():
        return "Please upload a lab report (PDF/image) or paste its text."
    findings = parse_values(text)
    if not findings:
        return ("Couldn't recognise any known lab markers in this report. "
                "Try pasting the text, or check the sample format in /samples.")
    try:
        result = call_model(findings)
    except Exception as e:
        # graceful fallback so a flaky endpoint never kills the demo
        result = {
            "overall_summary": "Model endpoint unavailable — showing the deterministic analysis only.",
            "findings": [
                {"marker": f["marker"], "value": f"{f['value']} {f['unit']}",
                 "status": f["status"], "plain_explanation": f"Reference range {f['reference']}."}
                for f in findings
            ],
            "recommended_specialist": next(
                (f["specialist"] for f in findings if f["status"] != "normal"),
                "General Practitioner"),
            "urgency": deterministic_urgency(findings),
            "urgency_reason": f"(LLM error: {e})",
        }
    return render(result, findings)


# --------------------------------------------------------------------------- #
# UI                                                                          #
# --------------------------------------------------------------------------- #
with gr.Blocks(title="MedLens") as demo:
    gr.Markdown(
        "# 🩸 MedLens\n"
        "Upload a lab report → understand what's off and which doctor to see.\n\n"
        "*Inference runs on an open model served on AMD (vLLM on AMD Developer Cloud / Fireworks).*"
    )
    with gr.Row():
        with gr.Column():
            file_in = gr.File(label="Lab report (PDF / PNG / JPG)",
                              file_types=[".pdf", ".png", ".jpg", ".jpeg", ".txt"],
                              type="filepath")
            text_in = gr.Textbox(label="…or paste the report text", lines=8,
                                 placeholder="Hemoglobin: 14.2 g/dL\nFasting Glucose: 92 mg/dL\n...")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out = gr.Markdown()
    btn.click(analyze, inputs=[file_in, text_in], outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

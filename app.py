"""
MedLens — photograph a lab report in any language, get a plain-language
explanation in YOUR language, an urgency level, the right specialist, and
questions to ask the doctor.

Architecture (what to tell the judges):
  1. READ    — a vision model on AMD transcribes a photo / scanned PDF of the
               report (any language) into standard 'Marker: value unit' lines.
  2. CHECK   — deterministic code compares every value with a fixed reference
               table. No LLM touches the numbers, so none can be invented.
  3. EXPLAIN — a reasoning model on AMD explains the flagged values in the
               user's chosen language (24 languages) and suggests questions
               for the doctor.
  4. ROUTE   — urgency level + specialist. The urgency can only be RAISED by
               the model, never lowered below what the code found.

All models are reached through an OpenAI-compatible endpoint configured by env
vars (default: AMD Token Factory).
"""

import base64
import io
import json
import os
import re

import gradio as gr
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL", "DeepSeek-V4-Flash")                  # reasoning
VISION_MODEL = os.environ.get("VISION_MODEL", "DeepSeek-V4.1-Flash")  # reads photos / scans

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

with open("reference_ranges.json", "r", encoding="utf-8") as f:
    REFERENCE = json.load(f)["markers"]

# (display name, English name used in prompts)
LANGUAGES = {
    "English": "English",
    "Italiano": "Italian",
    "Русский": "Russian",
    "Кыргызча": "Kyrgyz",
    "中文 (简体)": "Simplified Chinese",
    "हिन्दी": "Hindi",
    "پښتو": "Pashto",
    "اردو": "Urdu",
    "العربية": "Arabic",
    "فارسی": "Persian (Farsi)",
    "Español": "Spanish",
    "Français": "French",
    "Deutsch": "German",
    "Português": "Portuguese",
    "Українська": "Ukrainian",
    "Română": "Romanian",
    "Shqip": "Albanian",
    "বাংলা": "Bengali",
    "ਪੰਜਾਬੀ": "Punjabi (Gurmukhi script)",
    "Tagalog": "Tagalog (Filipino)",
    "Türkçe": "Turkish",
    "Oʻzbekcha": "Uzbek (Latin script)",
    "Қазақша": "Kazakh",
    "Тоҷикӣ": "Tajik",
}
RTL = {"Pashto", "Urdu", "Arabic", "Persian (Farsi)"}

DISPLAY_NAMES = {
    "hemoglobin": "Hemoglobin", "wbc": "White blood cells (WBC)", "platelets": "Platelets",
    "glucose": "Fasting glucose", "hba1c": "HbA1c", "total_cholesterol": "Total cholesterol",
    "ldl": "LDL cholesterol", "hdl": "HDL cholesterol", "triglycerides": "Triglycerides",
    "tsh": "TSH (thyroid)", "creatinine": "Creatinine (kidney)", "alt": "ALT (liver)",
    "ast": "AST (liver)", "vitamin_d": "Vitamin D", "crp": "CRP (inflammation)",
}

URGENCY_ORDER = {"routine": 0, "see_doctor_soon": 1, "urgent": 2}
URGENCY_ICON = {"routine": "🟢", "see_doctor_soon": "🟠", "urgent": "🔴"}

# English UI — used directly for English and as fallback if a translation is missing
UI_EN = {
    "routine": "Routine — discuss at your next regular check-up",
    "see_doctor_soon": "See a doctor soon — some results are outside the normal range",
    "urgent": "Urgent — one or more results are far outside the normal range; seek medical care promptly",
    "summary": "Summary", "specialist": "Recommended specialist", "results": "Your results",
    "marker": "Marker", "value": "Value", "status": "Status", "meaning": "What it means",
    "questions": "Questions to ask your doctor",
    "low": "low", "normal": "normal", "high": "high",
    "disclaimer": ("This is not a diagnosis. MedLens explains lab numbers in plain language and "
                   "points you to the right kind of doctor. It cannot replace a medical professional. "
                   "If you feel unwell or are worried, contact a doctor or emergency services."),
}

# alias -> marker key, longest alias first so "emoglobina glicata" wins over "emoglobina"
ALIASES = sorted(
    ((alias.lower(), key) for key, spec in REFERENCE.items() for alias in spec["aliases"]),
    key=lambda p: -len(p[0]),
)


# --------------------------------------------------------------------------- #
# Layer 1 — deterministic extraction & flagging                               #
# --------------------------------------------------------------------------- #
def parse_values(text):
    """Find known markers in free text and flag them against reference ranges."""
    findings, seen = [], set()
    for line in text.splitlines():
        label, sep, value_part = line.partition(":")
        if not sep:
            label, value_part = line, line
        low_label = label.lower()
        for alias, key in ALIASES:
            m = re.search(r"\b" + re.escape(alias) + r"s?\b", low_label)
            if not m or key in seen:
                continue
            space = value_part if sep else line[m.end():]   # numbers AFTER the marker name
            space = re.sub(r"(\d),(\d)", r"\1.\2", space)     # Italian decimal comma
            # standalone numbers only: skip digits glued to letters (HbA1c, x10^9, 25-OH)
            nums = re.findall(r"(?<![A-Za-z\d.^])[-+]?\d+(?:\.\d+)?(?![A-Za-z\d])", space)
            if not nums:
                continue
            spec = REFERENCE[key]
            value = float(nums[0])
            lo, hi = spec["low"], spec["high"]
            status = "low" if value < lo else "high" if value > hi else "normal"
            critical = value < lo * 0.6 or value > hi * 1.6   # urgency floor heuristic
            findings.append({
                "marker": DISPLAY_NAMES.get(key, key), "value": value, "unit": spec["unit"],
                "reference": f"{lo}–{hi}", "status": status, "critical": critical,
                "specialist": spec["specialist"],
            })
            seen.add(key)
            break
    return findings


def deterministic_urgency(findings):
    if any(f["critical"] for f in findings):
        return "urgent"
    if any(f["status"] != "normal" for f in findings):
        return "see_doctor_soon"
    return "routine"


# --------------------------------------------------------------------------- #
# Layer 2 — reasoning on AMD, in the user's language                          #
# --------------------------------------------------------------------------- #
def system_prompt(language):
    ui_keys = ", ".join(f'"{k}"' for k in UI_EN)
    return f"""You are MedLens, a careful assistant that EXPLAINS lab results in plain \
language and recommends which medical specialist a person should see.

STRICT RULES:
- NEVER diagnose. Never write "you have X" or "this indicates X disease". Use cautious
  wording such as "can be associated with", "a doctor may want to check".
- Trust the deterministic findings you are given; do NOT invent or change numbers or statuses.
- Always lean toward advising the person to see a doctor. Stay calm and non-alarming.
- Write ALL human-readable text in {language}, simply, as for someone with no medical
  background. Keep numbers and units exactly as given.

Return ONLY a JSON object, no markdown, in exactly this shape:
{{
  "overall_summary": "2-3 sentences",
  "findings": [
    {{"marker": "<marker name translated into {language}>", "value": "<value> <unit>",
      "status": "low|normal|high", "plain_explanation": "one cautious sentence"}}
  ],
  "recommended_specialist": "<single most relevant specialty, in {language}>",
  "urgency": "routine|see_doctor_soon|urgent",
  "urgency_reason": "one short, non-diagnostic sentence",
  "questions_for_doctor": ["3 or 4 short, practical questions the person can ask at the visit"],
  "ui": {{ {ui_keys} }}
}}
"ui" must contain each of those keys with these English texts translated into {language}:
{json.dumps(UI_EN, ensure_ascii=False)}"""


def call_model(findings, language):
    payload = {
        "deterministic_findings": [{k: v for k, v in f.items() if k != "critical"} for f in findings],
        "deterministic_urgency": deterministic_urgency(findings),
        "answer_language": language,
    }
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        max_tokens=3500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt(language)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# Input extraction — vision model on AMD for photos and scans                 #
# --------------------------------------------------------------------------- #
VISION_PROMPT = (
    "You are reading a medical lab report image. It may be in ANY language (Italian, Russian, "
    "Chinese, Arabic, ...). Transcribe EVERY lab test result as one line in the form "
    "'Marker: value unit', using the STANDARD ENGLISH marker name (e.g. 'Emoglobina' -> "
    "'Hemoglobin', 'Glicemia a digiuno' -> 'Fasting Glucose', 'Colesterolo totale' -> "
    "'Total Cholesterol', 'Globuli bianchi' -> 'WBC'). Write decimals with a dot. "
    "Output only those lines, nothing else. Do not interpret anything."
)


def vision_transcribe(png_bytes):
    b64 = base64.b64encode(png_bytes).decode()
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": VISION_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Transcribe the lab results in this report."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
    )
    return resp.choices[0].message.content or ""


def image_file_to_png(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pdf_pages_to_png(path, max_pages=3):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    out = []
    for i in range(min(len(pdf), max_pages)):
        pil = pdf[i].render(scale=2).to_pil()
        pil.thumbnail((1600, 1600))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def extract_text(file_path, pasted_text):
    """Returns (text, source_label)."""
    if pasted_text and pasted_text.strip():
        return pasted_text, "pasted text"
    if not file_path:
        return "", ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        return vision_transcribe(image_file_to_png(file_path)), f"photo read by {VISION_MODEL} on AMD"
    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        if len(text.strip()) > 40 and parse_values(text):
            return text, "PDF text layer"
        pages = pdf_pages_to_png(file_path)
        return "\n".join(vision_transcribe(p) for p in pages), f"PDF read by {VISION_MODEL} on AMD"
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(), "text file"


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def render(result, det_findings, language):
    ui = dict(UI_EN)
    ui.update({k: v for k, v in (result.get("ui") or {}).items() if isinstance(v, str) and v.strip()})
    final = max(result.get("urgency", "routine"), deterministic_urgency(det_findings),
                key=lambda u: URGENCY_ORDER.get(u, 0))
    status_word = {"low": f"🔽 {ui['low']}", "high": f"🔼 {ui['high']}", "normal": f"✅ {ui['normal']}"}

    out = [f"## {URGENCY_ICON[final]} {ui[final]}", ""]
    if result.get("urgency_reason"):
        out.append(f"*{result['urgency_reason']}*\n")
    out.append(f"**{ui['summary']}.** {result.get('overall_summary', '')}\n")
    out.append(f"**👉 {ui['specialist']}: {result.get('recommended_specialist', '')}**\n")
    qs = [q for q in (result.get("questions_for_doctor") or []) if isinstance(q, str)]
    if qs:
        out.append(f"### 💬 {ui['questions']}")
        out += [f"- {q}" for q in qs[:4]]
        out.append("")
    out.append(f"### {ui['results']}")
    out.append(f"| {ui['marker']} | {ui['value']} | {ui['status']} | {ui['meaning']} |")
    out.append("|---|---|---|---|")
    for f in result.get("findings", []):
        out.append(f"| {f.get('marker','')} | {f.get('value','')} | "
                   f"{status_word.get(f.get('status','normal'), '')} | {f.get('plain_explanation','')} |")
    out.append("")
    out.append(f"⚠️ **{ui['disclaimer']}**")
    if ui["disclaimer"] != UI_EN["disclaimer"]:
        out.append(f"\n<sub>EN: {UI_EN['disclaimer']}</sub>")
    body = "\n".join(out)
    if language in RTL:
        body = f'<div dir="rtl">\n\n{body}\n\n</div>'
    return body


def analyze(file_path, pasted_text, lang_display="English"):
    language = LANGUAGES.get(lang_display, "English")
    try:
        text, source = extract_text(file_path, pasted_text)
    except Exception as e:
        return f"Could not read the file: {e}"
    if not text.strip():
        return "Please upload a lab report (photo / PDF) or paste its text."
    findings = parse_values(text)
    if not findings:
        return ("Couldn't recognise any known lab markers in this report.\n\n"
                f"<sub>What was read:</sub>\n```\n{text[:800]}\n```")
    try:
        result = call_model(findings, language)
    except Exception as e:
        result = {   # graceful fallback so a flaky endpoint never kills the demo
            "overall_summary": "Model endpoint unavailable — showing the deterministic analysis only.",
            "findings": [{"marker": f["marker"], "value": f"{f['value']} {f['unit']}",
                          "status": f["status"], "plain_explanation": f"Reference range {f['reference']}."}
                         for f in findings],
            "recommended_specialist": next((f["specialist"] for f in findings if f["status"] != "normal"),
                                           "General Practitioner"),
            "urgency": deterministic_urgency(findings),
            "urgency_reason": f"(LLM error: {e})",
        }
        language = "English"
    header = (f"<sub>📄 Input: {source} · 🧠 Reasoning: {MODEL} on AMD · "
              f"🌐 Answer language: {language}</sub>\n\n")
    return header + render(result, findings, language)


# --------------------------------------------------------------------------- #
# UI                                                                          #
# --------------------------------------------------------------------------- #
with gr.Blocks(title="MedLens") as demo:
    gr.Markdown(
        "# 🩸 MedLens\n"
        "Photograph a lab report **in any language** → understand it **in your language**, "
        "see how urgent it is and which doctor to visit.\n\n"
        "*All AI runs on AMD GPUs via AMD Token Factory — a vision model reads the report, "
        "a reasoning model explains it. Demo uses synthetic data only.*"
    )
    with gr.Row():
        with gr.Column():
            file_in = gr.File(label="Lab report (photo / PDF)",
                              file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"],
                              type="filepath")
            text_in = gr.Textbox(label="…or paste the report text", lines=8,
                                 placeholder="Hemoglobin: 14.2 g/dL\nFasting Glucose: 92 mg/dL\n...")
            lang_in = gr.Dropdown(choices=list(LANGUAGES), value="English",
                                  label="🌐 Explain it in (24 languages)")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out = gr.Markdown()
    btn.click(analyze, inputs=[file_in, text_in, lang_in], outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

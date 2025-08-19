# streamlit_app_advanced.py
# -----------------------------------------------------------
# Multimodal GenAI for Solar Panel Inspection (Advanced)
# Adds: Validation layer, Evaluation metrics, OCR, and Video support
# -----------------------------------------------------------

import os
import re
import io
import json
import zipfile
import tempfile
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from jsonschema import validate, ValidationError
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.prompts import PromptTemplate
from openai import OpenAI

# -------- Optional / best-effort dependencies --------
OCR_BACKEND = None
try:
    import pytesseract  # requires tesseract installed on OS
    # OCR_BACKEND = "pytesseract"
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        OCR_BACKEND = "tesseract"
    else:
        raise FileNotFoundError("Tesseract binary not found at expected path.")
except Exception as e:
    ocr_error = str(e)
    # --- Fallback: EasyOCR ---
    try:
        import easyocr
        OCR_BACKEND = "easyocr"
        reader = easyocr.Reader(['en'])
    except Exception as ee:
        ocr_error += f" | EasyOCR also failed: {ee}"
        OCR_BACKEND = None

try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

# ----------------- ENV & CLIENT ----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Multimodal GenAI – Advanced", layout="wide")
st.title("⚡ Multimodal GenAI – Solar Panel Inspection ")
st.caption("BLIP + GPT + LangChain + OCR + Video + Validation/Eval")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to your environment or .env file.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------- MODELS (cache) --------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

BLIP_PROCESSOR, BLIP_MODEL = load_blip()

# ----------------- JSON SCHEMA -----------------------
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "Issue": {"type": "string", "minLength": 3},
        "Component": {"type": "string", "minLength": 2},
        "Severity": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "Recommendation": {"type": "string", "minLength": 3},
        "Timeline": {"type": "string", "minLength": 2},
    },
    "required": ["Issue", "Component", "Severity", "Recommendation", "Timeline"],
    "additionalProperties": True
}

# ----------------- HELPERS ---------------------------
def caption_image(image: Image.Image) -> str:
    inputs = BLIP_PROCESSOR(images=image, return_tensors="pt")
    output = BLIP_MODEL.generate(**inputs)
    return BLIP_PROCESSOR.decode(output[0], skip_special_tokens=True)

def ocr_image(image: Image.Image) -> str:
    """Return extracted text from image using available OCR backend."""
    # if OCR_BACKEND == "pytesseract":
    #     try:
    #         return pytesseract.image_to_string(image)
    #     except Exception as e:
    #         return f"[OCR-error: {e}]"
    # elif OCR_BACKEND == "easyocr":
    #     try:
    #         reader = easyocr.Reader(['en'], gpu=False)
    #         result = reader.readtext(np.array(image))
    #         # Concatenate text strings
    #         return " ".join([seg[1] for seg in result]) if result else ""
    #     except Exception as e:
    #         return f"[OCR-error: {e}]"
    # else:
    #     return ""  # OCR not available
    
    global OCR_BACKEND
    if OCR_BACKEND == "tesseract":
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            return f"[OCR-error: {e}]"
    elif OCR_BACKEND == "easyocr":
        try:
            results = reader.readtext(np.array(image), detail=0)
            return " ".join(results)
        except Exception as e:
            return f"[OCR-error: {e}]"
    else:
        return f"[OCR-error: No OCR backend available ({ocr_error})]"

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Try to robustly extract the first JSON object from a text block."""
    # Try simple json parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to locate JSON object with regex
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None

def validate_report(report_text: str) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Validate LLM output against schema. Returns (is_valid, errors, parsed_json)."""
    parsed = extract_json_from_text(report_text)
    if parsed is None:
        return False, ["Could not parse JSON from model output."], None
    try:
        validate(instance=parsed, schema=REPORT_SCHEMA)
        return True, [], parsed
    except ValidationError as e:
        return False, [str(e)], parsed

def eval_report(parsed: Dict[str, Any], caption: str, tech_note: str, ocr_text: str) -> Dict[str, Any]:
    """
    Simple rule-based evaluation metrics:
      - completeness: required keys present & non-empty
      - severity_valid: in allowed set
      - length_quality: checks minimal lengths
      - keyword_overlap: basic overlap between (caption+note+ocr) and Issue/Recommendation
      - consistency_flags: naive heuristics, e.g., if Issue mentions 'crack' but severity 'Low'
    Produces overall score (0-100).
    """
    metrics = {}
    required = ["Issue", "Component", "Severity", "Recommendation", "Timeline"]
    missing = [k for k in required if k not in parsed or not str(parsed.get(k, "")).strip()]
    metrics["required_fields_present"] = (len(missing) == 0)
    metrics["missing_fields"] = missing

    sev = str(parsed.get("Severity", "")).strip()
    metrics["severity_valid"] = sev in ["Low", "Medium", "High"]

    # Length checks
    def _len_ok(x, n=5): return len(str(x).strip()) >= n
    len_checks = {k: _len_ok(parsed.get(k, ""), 5) for k in ["Issue", "Recommendation"]}
    metrics["length_quality"] = all(len_checks.values())

    # Keyword overlap
    source_text = f"{caption} {tech_note} {ocr_text}".lower()
    issue_text = str(parsed.get("Issue", "")).lower()
    rec_text = str(parsed.get("Recommendation", "")).lower()

    def overlap_score(a: str, b: str) -> float:
        ta = set(re.findall(r"[a-z0-9]+", a))
        tb = set(re.findall(r"[a-z0-9]+", b))
        if not ta or not tb:
            return 0.0
        return 100.0 * len(ta & tb) / max(1, len(tb))

    metrics["keyword_overlap_issue"] = round(overlap_score(source_text, issue_text), 1)
    metrics["keyword_overlap_recommendation"] = round(overlap_score(source_text, rec_text), 1)

    # Consistency flags (very naive)
    flags = []
    if "crack" in issue_text and sev == "Low":
        flags.append("Issue mentions crack but Severity is Low.")
    if "replace" in rec_text and sev == "Low":
        flags.append("Recommendation suggests replacement while Severity is Low.")
    metrics["consistency_flags"] = flags

    # Aggregate score (heuristic)
    score = 0
    score += 40 if metrics["required_fields_present"] else max(0, 40 - 10 * len(missing))
    score += 15 if metrics["severity_valid"] else 0
    score += 15 if metrics["length_quality"] else 0
    score += min(15, metrics["keyword_overlap_issue"] * 0.15)  # cap 15
    score += min(15, metrics["keyword_overlap_recommendation"] * 0.15)
    score = int(max(0, min(100, round(score))))
    metrics["overall_score"] = score
    return metrics

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["caption", "tech_note", "ocr_text", "frame_captions"],
    template="""
You are a solar panel inspection analyst. Using the following multimodal evidence, produce a strict JSON report with fields:
Issue, Component, Severity (Low/Medium/High), Recommendation, Timeline.

- Image Caption (representative): {caption}
- Technician Note: {tech_note}
- OCR Text (if any): {ocr_text}
- Video Frame Captions (if any): {frame_captions}

Rules:
1) Output valid JSON only, no extra commentary.
2) Severity must be one of: Low, Medium, High.
3) Align recommendations with severity and evidence.
4) Be specific about component/location when possible.
"""
)

def llm_generate_report(model_name: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an AI inspection assistant. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

def normalize_image_name(path: str) -> str:
    return os.path.basename(path).lower()

# ----------------- SIDEBAR ---------------------------
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("LLM Model", ["gpt-4", "gpt-3.5-turbo"], index=0)
    enable_ocr = st.checkbox("Enable OCR", value=True)
    show_validation = st.checkbox("Show Validation & Metrics", value=True)
    st.markdown("---")
    st.caption(f"OCR backend: **{OCR_BACKEND or 'None'}**, OpenCV: **{'OK' if CV2_OK else 'Unavailable'}**")

# =====================================================
#                TABS: Single | Batch | Video
# =====================================================
tab_single, tab_batch, tab_video = st.tabs(["🖼 Single Image", "📦 Batch ZIP", "🎞 Video"])

# ----------------- SINGLE IMAGE ----------------------
with tab_single:
    st.subheader("Single Image → Caption → JSON Report")
    col_left, col_right = st.columns([1,1])

    with col_left:
        img_file = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        tech_note = st.text_area("Technician Note", placeholder="Describe what the technician observed...")

        ocr_text = ""
        if img_file:
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="Input Image", use_container_width=True)
            if enable_ocr:
                with st.spinner("Running OCR..."):
                    ocr_text = ocr_image(image)
                if ocr_text.strip():
                    st.info(f"**OCR Text:** {ocr_text[:400]}{'...' if len(ocr_text)>400 else ''}")
                else:
                    st.caption("No OCR text detected (or OCR disabled/unavailable).")

    with col_right:
        if st.button("Generate Report (Single)"):
            if not img_file or not tech_note.strip():
                st.error("Please upload an image and enter a technician note.")
            else:
                with st.spinner("Captioning with BLIP..."):
                    caption = caption_image(image)
                    st.success(f"Caption: {caption}")

                prompt = PROMPT_TEMPLATE.format(
                    caption=caption,
                    tech_note=tech_note,
                    ocr_text=ocr_text if enable_ocr else "",
                    frame_captions=""  # N/A for single image
                )

                with st.spinner(f"Generating report with {model_name}..."):
                    report_text = llm_generate_report(model_name, prompt)

                st.subheader("Structured Inspection Report (Model Output)")
                st.code(report_text, language="json")

                if show_validation:
                    valid, errors, parsed = validate_report(report_text)
                    if valid:
                        st.success("✅ Report is VALID against schema.")
                    else:
                        st.error("❌ Report failed validation.")
                        for e in errors:
                            st.write(f"- {e}")

                    if parsed:
                        metrics = eval_report(parsed, caption, tech_note, ocr_text)
                        st.subheader("Evaluation Metrics")
                        st.json(metrics)

# ----------------- BATCH ZIP -------------------------
with tab_batch:
    st.subheader("Batch ZIP: Images + technician_notes.csv")
    st.caption("CSV columns required: `image_name`, `technical_note` (case-insensitive filename match).")
    zip_up = st.file_uploader("Upload ZIP", type=["zip"])

    if zip_up:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_up.read())
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)

            # Recursively gather files
            paths = []
            for root, _, files in os.walk(tmp_dir):
                for f in files:
                    paths.append(os.path.join(root, f))

            images = sorted([p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))])
            csvs = [p for p in paths if p.lower().endswith(".csv")]

            if not csvs:
                st.error("No CSV found. Ensure your ZIP includes technician_notes.csv.")
            else:
                df = pd.read_csv(csvs[0])
                if not set(["image_name", "technical_note"]).issubset(df.columns):
                    st.error("CSV must include 'image_name' and 'technical_note' columns.")
                else:
                    df["image_name_lower"] = df["image_name"].str.lower()

                    results = []
                    for img_path in images:
                        try:
                            img = Image.open(img_path).convert("RGB")
                            img_name = normalize_image_name(img_path)
                            row = df[df["image_name_lower"] == img_name]
                            if row.empty:
                                st.warning(f"No technician note for {img_name}. Skipping.")
                                continue

                            note = row.iloc[0]["technical_note"]

                            if enable_ocr:
                                ocr_txt = ocr_image(img)
                            else:
                                ocr_txt = ""

                            with st.spinner(f"Captioning {os.path.basename(img_path)} ..."):
                                cap = caption_image(img)

                            prompt = PROMPT_TEMPLATE.format(
                                caption=cap,
                                tech_note=note,
                                ocr_text=ocr_txt,
                                frame_captions=""
                            )
                            report_text = llm_generate_report(model_name, prompt)

                            valid, errors, parsed = validate_report(report_text)
                            metrics = eval_report(parsed, cap, note, ocr_txt) if parsed else {}

                            # UI visualization per item
                            with st.expander(f"📄 {os.path.basename(img_path)}", expanded=False):
                                st.image(img, caption=os.path.basename(img_path), use_container_width=True)
                                st.markdown(f"**Technician Note:** {note}")
                                st.markdown(f"**BLIP Caption:** {cap}")
                                if enable_ocr and ocr_txt:
                                    st.caption(f"**OCR:** {ocr_txt[:300]}{'...' if len(ocr_txt)>300 else ''}")
                                st.markdown("**Model Report:**")
                                st.code(report_text, language="json")
                                if show_validation:
                                    if valid:
                                        st.success("VALID report ✅")
                                    else:
                                        st.error("Invalid report ❌")
                                        for e in errors: st.write(f"- {e}")
                                    st.json(metrics)

                            results.append({
                                "image_name": os.path.basename(img_path),
                                "caption": cap,
                                "technician_note": note,
                                "ocr_text": ocr_txt,
                                "report_raw": report_text,
                                "valid": valid,
                                "errors": "; ".join(errors) if errors else "",
                                "overall_score": (metrics.get("overall_score") if metrics else None)
                            })
                        except Exception as e:
                            st.error(f"Error processing {img_path}: {e}")

                    if results:
                        out_df = pd.DataFrame(results)
                        st.success("Batch completed.")
                        st.dataframe(out_df, use_container_width=True)
                        st.download_button(
                            "📥 Download Batch Results (CSV)",
                            data=out_df.to_csv(index=False).encode(),
                            file_name="batch_inspection_results.csv",
                            mime="text/csv"
                        )

# ----------------- VIDEO -----------------------------
with tab_video:
    st.subheader("Video → Sample Frames → Captions → Aggregated Report")
    if not CV2_OK:
        st.warning("OpenCV is not available in this environment. Install opencv-python to enable video mode.")
    else:
        vid_file = st.file_uploader("Upload video (MP4/AVI/MOV)", type=["mp4", "avi", "mov"])
        vid_note = st.text_area("Technician Note (Video Context)", placeholder="Notes relevant to this video...")
        frame_every_s = st.slider("Sample a frame every N seconds", min_value=1, max_value=10, value=2, step=1)

        if st.button("Process Video"):
            if not vid_file or not vid_note.strip():
                st.error("Upload a video and enter a technician note.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpv:
                    tmpv.write(vid_file.read())
                    video_path = tmpv.name

                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                interval = max(1, int(fps * frame_every_s))

                frame_captions: List[str] = []
                ocr_snippets: List[str] = []
                frames_collected = 0
                idx = 0

                with st.spinner("Sampling frames and generating captions..."):
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if idx % interval == 0:
                            frames_collected += 1
                            # Convert frame to PIL.Image
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil = Image.fromarray(rgb)

                            # Caption
                            cap_text = caption_image(pil)
                            frame_captions.append(cap_text)

                            # OCR
                            if enable_ocr:
                                txt = ocr_image(pil)
                                if txt.strip():
                                    ocr_snippets.append(txt[:200])
                        idx += 1
                cap.release()

                # Aggregate captions + OCR
                representative_caption = frame_captions[0] if frame_captions else "No frames sampled."
                frame_caps_joined = "; ".join(frame_captions[:10])  # limit prompt size
                ocr_agg = " ".join(ocr_snippets[:10])

                st.info(f"Frames sampled: {frames_collected}")
                st.markdown("**Sampled Frame Captions (subset):**")
                if frame_captions:
                    st.write(frame_captions[:10])
                if enable_ocr and ocr_agg:
                    st.caption(f"Aggregated OCR (subset): {ocr_agg[:300]}{'...' if len(ocr_agg)>300 else ''}")

                prompt = PROMPT_TEMPLATE.format(
                    caption=representative_caption,
                    tech_note=vid_note,
                    ocr_text=ocr_agg,
                    frame_captions=frame_caps_joined
                )

                with st.spinner(f"Generating aggregated report with {model_name}..."):
                    report_text = llm_generate_report(model_name, prompt)

                st.subheader("Aggregated Video Inspection Report")
                st.code(report_text, language="json")

                if show_validation:
                    valid, errors, parsed = validate_report(report_text)
                    if valid:
                        st.success("✅ Report is VALID against schema.")
                    else:
                        st.error("❌ Report failed validation.")
                        for e in errors:
                            st.write(f"- {e}")

                    if parsed:
                        metrics = eval_report(parsed, representative_caption, vid_note, ocr_agg)
                        st.subheader("Evaluation Metrics")
                        st.json(metrics)

st.markdown("---")
st.caption("Tip: If Tesseract OCR is installed system-wide, pytesseract will be used. Otherwise EasyOCR is attempted. Video mode requires OpenCV.")

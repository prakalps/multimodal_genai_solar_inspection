# streamlit_app_advanced.py
# -----------------------------------------------------------
# Multimodal GenAI for Solar Panel Inspection (Advanced)
# + Analytics dashboard
# + Caption model flexibility (BLIP-base, BLIP-large, ViT-GPT2)
# + Validation layer, Evaluation metrics, OCR, and Video support
# -----------------------------------------------------------

import os, re, io, json, zipfile, tempfile, time, random, hashlib
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from jsonschema import validate, ValidationError
from langchain.prompts import PromptTemplate
from openai import OpenAI

# Charts
import matplotlib.pyplot as plt

# Optional fuzzy matching for batch mode
try:
    from rapidfuzz import process as rf_process
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# -------- Optional / best-effort dependencies --------
OCR_BACKEND, ocr_error = None, None
try:
    import pytesseract  # Python wrapper; needs system tesseract
    # Point to Windows default install if present; adjust as needed.
    TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(TESSERACT_EXE):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
        OCR_BACKEND = "tesseract"
    else:
        raise FileNotFoundError("Tesseract binary not found at expected path.")
except Exception as e:
    ocr_error = str(e)
    try:
        import easyocr
        OCR_BACKEND = "easyocr"
        _EASY_READER = easyocr.Reader(['en'], gpu=False)
    except Exception as ee:
        ocr_error = f"{ocr_error} | EasyOCR also failed: {ee}"
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
st.title("⚡ Multimodal GenAI – Solar Panel Inspection (Advanced)")
st.caption("BLIP/ViT-GPT2 + GPT + LangChain + OCR + Video + Validation/Eval + Analytics")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to your environment or .env file.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------- CAPTION MODELS (flex) -------------
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

@st.cache_resource(show_spinner=True)
def load_blip(model_name: str):
    proc = BlipProcessor.from_pretrained(model_name)
    mdl = BlipForConditionalGeneration.from_pretrained(model_name)
    mdl.eval()
    return ("blip", proc, mdl)

@st.cache_resource(show_spinner=True)
def load_vit_gpt2():
    # Import lazily to avoid extra load if not selected
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    try:
        fe_cls = ViTImageProcessor
    except Exception:
        from transformers import ViTFeatureExtractor as ViTImageProcessor  # legacy fallback
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model.eval()
    return ("vitgpt2", feature_extractor, tokenizer, model)

def get_captioner(choice: str):
    """
    Returns a callable: caption_fn(image: PIL.Image) -> str
    """
    if choice == "BLIP base (Salesforce/blip-image-captioning-base)":
        mode, proc, mdl = load_blip("Salesforce/blip-image-captioning-base")
        def _cap(img):
            inputs = proc(images=img, return_tensors="pt")
            with torch.no_grad():
                out = mdl.generate(**inputs, max_length=40)
            return proc.decode(out[0], skip_special_tokens=True)
        return _cap

    if choice == "BLIP large (Salesforce/blip-image-captioning-large)":
        mode, proc, mdl = load_blip("Salesforce/blip-image-captioning-large")
        def _cap(img):
            inputs = proc(images=img, return_tensors="pt")
            with torch.no_grad():
                out = mdl.generate(**inputs, max_length=40)
            return proc.decode(out[0], skip_special_tokens=True)
        return _cap

    # ViT-GPT2 lightweight option
    mode, fe, tok, mdl = load_vit_gpt2()
    def _cap(img):
        pixel_values = fe(images=img, return_tensors="pt").pixel_values
        with torch.no_grad():
            output_ids = mdl.generate(pixel_values, max_length=24, num_beams=4)
        return tok.decode(output_ids[0], skip_special_tokens=True).strip()
    return _cap

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
def ocr_image(image: Image.Image) -> str:
    global OCR_BACKEND
    if OCR_BACKEND == "tesseract":
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            return f"[OCR-error: {e}]"
    elif OCR_BACKEND == "easyocr":
        try:
            results = _EASY_READER.readtext(np.array(image), detail=0)
            return " ".join(results)
        except Exception as e:
            return f"[OCR-error: {e}]"
    else:
        return f"[OCR-error: No OCR backend available ({ocr_error})]"

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def validate_report(report_text: str) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    parsed = extract_json_from_text(report_text)
    if parsed is None:
        return False, ["Could not parse JSON from model output."], None
    try:
        validate(instance=parsed, schema=REPORT_SCHEMA)
        return True, [], parsed
    except ValidationError as e:
        return False, [str(e)], parsed

def eval_report(parsed: Dict[str, Any], caption: str, tech_note: str, ocr_text: str) -> Dict[str, Any]:
    metrics = {}
    required = ["Issue", "Component", "Severity", "Recommendation", "Timeline"]
    missing = [k for k in required if k not in parsed or not str(parsed.get(k, "")).strip()]
    metrics["required_fields_present"] = (len(missing) == 0)
    metrics["missing_fields"] = missing
    sev = str(parsed.get("Severity", "")).strip()
    metrics["severity_valid"] = sev in ["Low", "Medium", "High"]

    def _len_ok(x, n=5): return len(str(x).strip()) >= n
    length_ok = all(_len_ok(parsed.get(k, ""), 5) for k in ["Issue", "Recommendation"])
    metrics["length_quality"] = length_ok

    source = f"{caption} {tech_note} {ocr_text}".lower()
    issue_text = str(parsed.get("Issue", "")).lower()
    rec_text = str(parsed.get("Recommendation", "")).lower()

    def overlap(a: str, b: str) -> float:
        ta = set(re.findall(r"[a-z0-9]+", a))
        tb = set(re.findall(r"[a-z0-9]+", b))
        if not ta or not tb: return 0.0
        return 100.0 * len(ta & tb) / max(1, len(tb))

    metrics["keyword_overlap_issue"] = round(overlap(source, issue_text), 1)
    metrics["keyword_overlap_recommendation"] = round(overlap(source, rec_text), 1)

    flags = []
    if "crack" in issue_text and sev == "Low":
        flags.append("Issue mentions crack but Severity is Low.")
    if "replace" in rec_text and sev == "Low":
        flags.append("Recommendation suggests replacement while Severity is Low.")
    metrics["consistency_flags"] = flags

    score = 0
    score += 40 if metrics["required_fields_present"] else max(0, 40 - 10 * len(missing))
    score += 15 if metrics["severity_valid"] else 0
    score += 15 if metrics["length_quality"] else 0
    score += min(15, metrics["keyword_overlap_issue"] * 0.15)
    score += min(15, metrics["keyword_overlap_recommendation"] * 0.15)
    metrics["overall_score"] = int(max(0, min(100, round(score))))
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

def with_retry(fn, attempts=4, base=0.8):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(base * (2 ** i) + random.random() * 0.2)

def llm_generate_report(model_name: str, prompt: str) -> str:
    def _call():
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an AI inspection assistant. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    return with_retry(_call)

def normalize_image_name(path: str) -> str:
    return os.path.basename(path).lower()

# ----------------- SIDEBAR ---------------------------
with st.sidebar:
    st.header("Settings")
    caption_choice = st.selectbox(
        "Caption Model",
        [
            "BLIP base (Salesforce/blip-image-captioning-base)",
            "BLIP large (Salesforce/blip-image-captioning-large)",
            "ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning)",
        ],
        index=0,
        help="Choose image captioning model. BLIP-large is slower but stronger; ViT-GPT2 is lightweight."
    )
    model_name = st.selectbox("LLM Model", ["gpt-4", "gpt-3.5-turbo"], index=0)
    enable_ocr = st.checkbox("Enable OCR", value=True)
    show_validation = st.checkbox("Show Validation & Metrics", value=True)
    st.markdown("---")
    st.caption(f"OCR backend: **{OCR_BACKEND or 'None'}**, OpenCV: **{'OK' if CV2_OK else 'Unavailable'}**")

caption_fn = get_captioner(caption_choice)

# Session store for analytics
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dict rows (same as batch results rows)

# =====================================================
#            TABS: Single | Batch | Video | Analytics
# =====================================================
tab_single, tab_batch, tab_video, tab_analytics = st.tabs(["🖼 Single Image", "📦 Batch ZIP", "🎞 Video", "📈 Analytics"])

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
                with st.spinner(f"Captioning with {caption_choice}..."):
                    caption = caption_fn(image)
                    st.success(f"Caption: {caption}")

                prompt = PROMPT_TEMPLATE.format(
                    caption=caption,
                    tech_note=tech_note,
                    ocr_text=ocr_text if enable_ocr else "",
                    frame_captions=""
                )

                with st.spinner(f"Generating report with {model_name}..."):
                    report_text = llm_generate_report(model_name, prompt)

                st.subheader("Structured Inspection Report (Model Output)")
                st.code(report_text, language="json")

                parsed, metrics, valid = None, {}, False
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

                # Push to session history for analytics
                row = {
                    "image_name": getattr(img_file, "name", "uploaded_image"),
                    "caption_model": caption_choice,
                    "caption": caption,
                    "technician_note": tech_note,
                    "ocr_text": ocr_text if enable_ocr else "",
                    "report_raw": report_text,
                    "valid": valid,
                    "overall_score": metrics.get("overall_score") if metrics else None,
                    "severity": (parsed or {}).get("Severity") if parsed else None
                }
                st.session_state["history"].append(row)
                st.success("Added to session analytics.")

# ----------------- BATCH ZIP -------------------------
def _match_note(img_name: str, df: pd.DataFrame) -> Optional[str]:
    # exact lowercase match
    row = df[df["image_name_lower"] == img_name]
    if not row.empty:
        return row.iloc[0]["technical_note"]
    if RAPIDFUZZ_OK:
        # fuzzy match fallback
        best = rf_process.extractOne(img_name, df["image_name_lower"].tolist(), score_cutoff=90)
        if best:
            match_value = best[0]
            row = df[df["image_name_lower"] == match_value]
            if not row.empty:
                return row.iloc[0]["technical_note"]
    return None

with tab_batch:
    st.subheader("Batch ZIP: Images + technician_notes.csv")
    st.caption("CSV columns required: `image_name`, `technical_note` (case-insensitive; fuzzy match enabled when needed).")
    zip_up = st.file_uploader("Upload ZIP", type=["zip"])

    if zip_up:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_up.read())
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)

            paths = [os.path.join(r, f) for r, _, files in os.walk(tmp_dir) for f in files]
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

                            note = _match_note(img_name, df)
                            if note is None:
                                st.warning(f"No technician note for {img_name}. Skipping.")
                                continue

                            ocr_txt = ocr_image(img) if enable_ocr else ""

                            with st.spinner(f"Captioning {os.path.basename(img_path)} with {caption_choice} ..."):
                                cap = caption_fn(img)

                            prompt = PROMPT_TEMPLATE.format(
                                caption=cap, tech_note=note, ocr_text=ocr_txt, frame_captions=""
                            )
                            report_text = llm_generate_report(model_name, prompt)

                            valid, errors, parsed = validate_report(report_text)
                            metrics = eval_report(parsed, cap, note, ocr_txt) if parsed else {}
                            severity = parsed.get("Severity") if parsed else None

                            with st.expander(f"📄 {os.path.basename(img_path)}", expanded=False):
                                st.image(img, caption=os.path.basename(img_path), use_container_width=True)
                                st.markdown(f"**Technician Note:** {note}")
                                st.markdown(f"**BLIP Caption:** {cap}")
                                if enable_ocr and ocr_txt:
                                    st.caption(f"**OCR:** {ocr_txt[:300]}{'...' if len(ocr_txt)>300 else ''}")
                                st.markdown("**Model Report:**")
                                st.code(report_text, language="json")
                                if show_validation:
                                    if valid: st.success("VALID report ✅")
                                    else:
                                        st.error("Invalid report ❌")
                                        for e in errors: st.write(f"- {e}")
                                    st.json(metrics)

                            row = {
                                "image_name": os.path.basename(img_path),
                                "caption_model": caption_choice,
                                "caption": cap,
                                "technician_note": note,
                                "ocr_text": ocr_txt,
                                "report_raw": report_text,
                                "valid": valid,
                                "overall_score": metrics.get("overall_score") if metrics else None,
                                "severity": severity
                            }
                            results.append(row)
                            st.session_state["history"].append(row)

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
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil = Image.fromarray(rgb)
                            cap_text = caption_fn(pil)
                            frame_captions.append(cap_text)
                            if enable_ocr:
                                txt = ocr_image(pil)
                                if txt.strip():
                                    ocr_snippets.append(txt[:200])
                        idx += 1
                cap.release()

                representative_caption = frame_captions[0] if frame_captions else "No frames sampled."
                frame_caps_joined = "; ".join(frame_captions[:10])
                ocr_agg = " ".join(ocr_snippets[:10])

                st.info(f"Frames sampled: {frames_collected}")
                if frame_captions:
                    st.markdown("**Sampled Frame Captions (subset):**")
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

                parsed, metrics, valid = None, {}, False
                if show_validation:
                    valid, errors, parsed = validate_report(report_text)
                    if valid:
                        st.success("✅ Report is VALID against schema.")
                    else:
                        st.error("❌ Report failed validation.")
                        for e in errors: st.write(f"- {e}")

                    if parsed:
                        metrics = eval_report(parsed, representative_caption, vid_note, ocr_agg)
                        st.subheader("Evaluation Metrics")
                        st.json(metrics)

                # Add to session history (as one "video" row)
                st.session_state["history"].append({
                    "image_name": getattr(vid_file, "name", "video"),
                    "caption_model": caption_choice,
                    "caption": representative_caption,
                    "technician_note": vid_note,
                    "ocr_text": ocr_agg,
                    "report_raw": report_text,
                    "valid": valid,
                    "overall_score": metrics.get("overall_score") if metrics else None,
                    "severity": (parsed or {}).get("Severity") if parsed else None
                })

# ----------------- ANALYTICS -------------------------
def _parse_reports(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    # Extract fields from JSON if possible (Issue, Component, etc.)
    issues, comps, sevs, recs, times = [], [], [], [], []
    for r in df["report_raw"].fillna(""):
        parsed = extract_json_from_text(r)
        if parsed:
            issues.append(parsed.get("Issue"))
            comps.append(parsed.get("Component"))
            sevs.append(parsed.get("Severity"))
            recs.append(parsed.get("Recommendation"))
            times.append(parsed.get("Timeline"))
        else:
            issues.append(None); comps.append(None); sevs.append(None); recs.append(None); times.append(None)
    df["Issue_p"] = issues
    df["Component_p"] = comps
    df["Severity_p"] = sevs
    df["Recommendation_p"] = recs
    df["Timeline_p"] = times
    return df

with tab_analytics:
    st.subheader("Analytics & Quality Dashboard")
    hist_df = _parse_reports(st.session_state["history"])

    if hist_df.empty:
        st.info("No session data yet. Run Single/Batch/Video first.")
    else:
        # KPIs
        total = len(hist_df)
        valid_pct = 100.0 * (hist_df["valid"].fillna(False).sum() / total)
        avg_score = hist_df["overall_score"].dropna().mean() if "overall_score" in hist_df else None
        st.metric("Total items processed", total)
        st.metric("Valid JSON (%)", f"{valid_pct:.1f}%")
        if avg_score is not None:
            st.metric("Avg Overall Score", f"{avg_score:.1f}")

        # Severity distribution
        sev_counts = hist_df["Severity_p"].fillna("Unknown").value_counts().sort_index()
        fig1, ax1 = plt.subplots()
        sev_counts.plot(kind="bar", ax=ax1)
        ax1.set_title("Severity Distribution")
        ax1.set_xlabel("Severity")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        # Score histogram
        if "overall_score" in hist_df and hist_df["overall_score"].notna().any():
            fig2, ax2 = plt.subplots()
            hist_df["overall_score"].dropna().astype(int).plot(kind="hist", bins=10, ax=ax2)
            ax2.set_title("Overall Score Histogram")
            ax2.set_xlabel("Score")
            st.pyplot(fig2)

        # Caption model usage
        model_counts = hist_df["caption_model"].value_counts()
        fig3, ax3 = plt.subplots()
        model_counts.plot(kind="bar", ax=ax3)
        ax3.set_title("Caption Model Usage")
        ax3.set_xlabel("Model")
        ax3.set_ylabel("Count")
        st.pyplot(fig3)

        # Show table + download
        st.dataframe(hist_df[[
            "image_name","caption_model","valid","overall_score","Severity_p","Issue_p","Component_p","Recommendation_p","Timeline_p"
        ]], use_container_width=True)

        st.download_button(
            "📥 Download Session Analytics (CSV)",
            data=hist_df.to_csv(index=False).encode(),
            file_name="session_analytics.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Tip: Choose a caption model in the sidebar. If Tesseract OCR isn’t available, the app tries EasyOCR. Video mode requires OpenCV.")

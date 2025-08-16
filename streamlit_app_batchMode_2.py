import os
import zipfile
import tempfile
import streamlit as st
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.prompts import PromptTemplate
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load BLIP model and processor
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip()

# Streamlit UI
st.set_page_config(page_title="Batch Mode: Solar Panel Inspection", layout="wide")
st.title("📦 Batch: Multimodal GenAI for Solar Panel Inspection")
st.markdown("Upload a ZIP file containing solar panel drone images and a `technician_notes.csv` file to generate structured inspection reports.")

# Upload ZIP
zip_file = st.file_uploader("Upload ZIP file with images + CSV", type=["zip"])

if zip_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Recursively find all files
        files = []
        for root, _, filenames in os.walk(tmp_dir):
            for f in filenames:
                files.append(os.path.join(root, f))

        image_files = sorted([f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        csv_files = [f for f in files if f.lower().endswith(".csv")]

        if not csv_files:
            st.error("❌ No CSV file with technician notes found.")
        else:
            notes_df = pd.read_csv(csv_files[0])
            if "image_name" not in notes_df.columns or "technical_note" not in notes_df.columns:
                st.error("❌ CSV must contain columns: image_name, technical_note")
            else:
                notes_df["image_name_lower"] = notes_df["image_name"].str.lower()
                results = []

                for image_path in image_files:
                    try:
                        image = Image.open(image_path).convert("RGB")
                        img_name = os.path.basename(image_path).lower()

                        # Match image name with CSV
                        note_row = notes_df[notes_df["image_name_lower"] == img_name]
                        if note_row.empty:
                            st.warning(f"⚠️ No technician note for {img_name}. Skipping.")
                            continue

                        tech_note = note_row.iloc[0]["technical_note"]

                        # BLIP image captioning
                        inputs = processor(images=image, return_tensors="pt")
                        output = blip_model.generate(**inputs)
                        caption = processor.decode(output[0], skip_special_tokens=True)

                        # Prompt engineering
                        prompt_template = PromptTemplate(
                            input_variables=["caption", "tech_note"],
                            template="""
You are a solar panel inspection analyst. Based on the following data, generate a JSON inspection report with these fields:
Issue, Component, Severity (Low/Medium/High), Recommendation, Timeline.

Image Caption: {caption}
Technician Note: {tech_note}

Format:
{{
  "Issue": "...",
  "Component": "...",
  "Severity": "...",
  "Recommendation": "...",
  "Timeline": "..."
}}
"""
                        )
                        final_prompt = prompt_template.format(caption=caption, tech_note=tech_note)

                        # GPT-4 structured report
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are an AI inspection assistant."},
                                {"role": "user", "content": final_prompt}
                            ],
                            temperature=0.1,
                            max_tokens=350
                        )
                        report = response.choices[0].message.content.strip()
                        results.append((os.path.basename(image_path), caption, tech_note, report))

                        # ----------------- Visualization -----------------
                        with st.expander(f"📂 Result for {os.path.basename(image_path)}", expanded=False):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.image(image, caption=os.path.basename(image_path), use_container_width=True)
                            with col2:
                                st.markdown(f"**🛠 Technician Note:**\n\n{tech_note}")
                                st.markdown(f"**📷 BLIP Caption:**\n\n{caption}")
                                st.markdown("**📝 Structured Inspection Report:**")
                                st.code(report, language="json")

                    except Exception as e:
                        st.error(f"❌ Error processing {image_path}: {e}")

                # Export results
                if results:
                    df_out = pd.DataFrame(results, columns=["image_name", "caption", "technician_note", "inspection_report"])
                    st.success("✅ All available reports generated.")
                    st.download_button(
                        "📥 Download All Reports (CSV)",
                        data=df_out.to_csv(index=False).encode(),
                        file_name="inspection_reports.csv",
                        mime="text/csv"
                    )

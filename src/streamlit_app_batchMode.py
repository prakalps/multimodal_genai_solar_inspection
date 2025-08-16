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
st.set_page_config(page_title="Batch Mode: Solar Panel Inspection", layout="centered")
st.title("📦 Batch: Multimodal GenAI for Solar Panel Inspection")
st.markdown("Upload a ZIP file with drone images and a `technician_notes.csv` file to generate inspection reports.")

# Upload ZIP
zip_file = st.file_uploader("Upload ZIP", type=["zip"])

if zip_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extract uploaded ZIP
        zip_path = os.path.join(tmp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Find images and CSV
        files = []
        for root, _, filenames in os.walk(tmp_dir):
            for f in filenames:
                files.append(os.path.join(root, f))

        # files = os.listdir(tmp_dir)
        image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        csv_files = [f for f in files if f.lower().endswith(".csv")]
        

        if not csv_files:
            st.error("No CSV file with technician notes found.")
        else:
            # notes_df = pd.read_csv(os.path.join(tmp_dir, csv_files[0]))
            notes_df = pd.read_csv(csv_files[0])  # No need to rejoin with tmp_dir now


            results = []
            for image_name in image_files:
                try:
                    # Normalize file names to lowercase
                    img_name = os.path.basename(image_name).lower()
                    image_path = os.path.join(tmp_dir, image_name)
                    image = Image.open(image_path).convert("RGB")

                    # Get corresponding note
                    note_row = notes_df[notes_df["image_name"] == image_name]
                    if note_row.empty:
                        st.warning(f"No technician note for {image_name}. Skipping.")
                        continue
                    tech_note = note_row.iloc[0]["technical_note"]

                    # Step 1: Image Captioning
                    inputs = processor(images=image, return_tensors="pt")
                    output = blip_model.generate(**inputs)
                    caption = processor.decode(output[0], skip_special_tokens=True)

                    # Step 2: Prompt Engineering
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

                    # Step 3: Call GPT-4
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
                    results.append((image_name, report))

                except Exception as e:
                    st.error(f"Failed to process {image_name}: {e}")

            # Display all reports
            if results:
                st.subheader("📝 Inspection Reports")
                for image_name, report in results:
                    st.markdown(f"**Image:** `{image_name}`")
                    st.code(report, language="json")

                # Export to CSV
                df_out = pd.DataFrame(results, columns=["image_name", "inspection_report"])
                st.download_button(
                    "📥 Download CSV Report",
                    data=df_out.to_csv(index=False).encode(),
                    file_name="batch_inspection_reports.csv",
                    mime="text/csv"
                )

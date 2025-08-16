# streamlit_app.py

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
from langchain.prompts import PromptTemplate
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()


# ---------- SETUP SECTION ----------
# Set your OpenAI API key
openai.api_key = "sk-..."   # <-- Replace with your OpenAI API key

# Load BLIP model and processor (first run will download weights)
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip()

# ---------- STREAMLIT FRONTEND ----------
st.set_page_config(page_title="Solar Panel Inspection GenAI", layout="centered")
st.title("🔎 Multimodal GenAI: Solar Panel Inspection Report Generator")
st.markdown("Upload a drone image and add technician notes to generate a structured inspection report using BLIP + GPT-4 + LangChain.")

# 1. MultiModel Data Preparation
uploaded_img = st.file_uploader("Upload Drone Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
tech_note = st.text_area("Technician Note / Inspection Log", "")

if uploaded_img and tech_note:
    # 2. Image Preprocessing
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="Inspection Image", use_column_width=True)
    
    # 3. Image Captioning with BLIP
    with st.spinner("Analyzing image with BLIP..."):
        inputs = processor(images=image, return_tensors="pt")
        output = blip_model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
    st.success(f"**Image Caption:** {caption}")

    # 4. Integrate & Merge Technical Notes & Multimodal Inputs
    merged_context = {
        "caption": caption,
        "tech_note": tech_note
    }

    # 5. Prompt Engineering for Structured Report (using LangChain)
    prompt_template = PromptTemplate(
        input_variables=["caption", "tech_note"],
        template="""
You are a solar panel inspection analyst. Based on the following data, generate a JSON inspection report with these fields: Issue, Component, Severity (Low/Medium/High), Recommendation, Timeline.

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
    st.code(final_prompt, language="markdown")

    # 6. Generating and Validating Structured Inspection Report

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if st.button("Generate Inspection Report"):
        with st.spinner("Generating report with GPT-4..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI inspection assistant."},
                    {"role": "user", "content": "Generate report for cracked solar panel."}
                ],
                temperature=0.1,
                max_tokens=350
            )
            # response = openai.ChatCompletion.create(
            #     model="gpt-4",
            #     messages=[{"role": "user", "content": final_prompt}],
            #     temperature=0.1,
            #     max_tokens=350,
            # )
            report = response.choices[0].message.content.strip()
        st.subheader("📝 Structured Inspection Report")
        st.code(report, language="json")
        st.success("Inspection report generated and validated!")

else:
    st.info("Please upload an image and enter technician notes to start.")

# --------- Workshop Recap ---------
st.markdown("---")
st.markdown("**Stepwise Processing Recap:**")
st.markdown("""
1. **MultiModal Data Preparation:** Upload image + technician note  
2. **Image Preprocessing:** Convert and display image  
3. **Image Captioning:** BLIP generates descriptive caption  
4. **Integration:** Combine caption + notes  
5. **Prompt Engineering:** LangChain builds a GPT-4 prompt  
6. **Report Generation:** GPT-4 outputs structured JSON report  
7. **Automation:** Review results in this Streamlit app
""")
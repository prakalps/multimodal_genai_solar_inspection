# Multimodal GenAI for Solar Panel Inspection 🚀

This repository contains a **Streamlit-based application** that demonstrates how **multimodal generative AI** can automate solar panel inspections using **drone images** and **technician notes**.

The app integrates:
- **BLIP (Bootstrapping Language-Image Pretraining)** for image captioning  
- **OpenAI GPT-4** for structured inspection report generation  
- **LangChain** for prompt engineering  
- **Streamlit** for the interactive UI  

---

## 📂 Features
1. Upload **drone image(s)** of solar panels  
2. Add or upload **technician notes**  
3. BLIP auto-generates **descriptive captions** from images  
4. Captions + notes are merged into a multimodal input  
5. GPT-4 generates a **structured inspection report (JSON)**  
6. Supports **single image mode** and **batch ZIP mode**  
7. Downloadable inspection reports in JSON format  
8. User-friendly visualization of **image → notes → report**  

# 🔹 Validation & Evaluation

- JSON Schema Validation – ensures reports conform to expected schema:
- - Issue, Component, Severity, Recommendation, Timeline
- Heuristic Evaluation Metrics:
- - Completeness of required fields
- - Severity validity checks
- - Keyword overlap between evidence and model output
- - Consistency flags (e.g., crack → severity mismatch)

# 🔹 🔍 Analytics (New in v1)

- Severity Distribution – bar chart of Low/Medium/High across reports.
- Overall Score Histogram – distribution of evaluation scores (0–100).
- Keyword Overlap Heatmap – visualize textual alignment between inputs and generated reports.
- Aggregate Dashboard – summary analytics from batch/video modes.

# 🔹 ⚙️ Model Flexibility (New in v1)

- LLM Selection: Choose GPT model (gpt-4 or gpt-3.5-turbo).
- Caption Model Selection: Switch between HuggingFace captioning models (BLIP-base, BLIP2, GIT, etc.) in the sidebar.
- Easy to extend – just add HuggingFace model names to the dropdown.
---

## 🛠️ Installation

Clone the repository:
```bash
git clone https://github.com/your-username/multimodal-genai-solar-inspection.git
cd multimodal-genai-solar-inspection
```
### 🧱 Repo Structure
```.
├── streamlit_app.py            # Your working app (single & batch modes)
├── requirements.txt
├── README.md
├── .env                        # OPENAI_API_KEY=sk-... (not committed)
├── .gitignore
└── samples/
    ├── technician_notes.csv    # Example CSV for batch mode
    └── sample_batch.zip        # Optional: demo ZIP for workshop
```

### ⚙️ Prerequisites
- Python 3.9+ (recommended)
- A valid OpenAI API key (new SDK, project or user key)
- Internet access (model & weights download on first run)

### 🚀 Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate    # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔑 Setup API Keys

Set your **OpenAI API key** as an environment variable:
Create a .env file in the project root:

```bash
export OPENAI_API_KEY="your_api_key_here"   # Mac/Linux
setx OPENAI_API_KEY "your_api_key_here"     # Windows
OPENAI_API_KEY=sk-REPLACE-WITH-YOUR-KEY

```
---
## 🧰 Config & Environment
- Model names:
    - BLIP: Salesforce/blip-image-captioning-base
    - OpenAI: gpt-4 (fallback to gpt-3.5-turbo if 4 isn’t available)
- Change LLM easily (inside client.chat.completions.create):

```model="gpt-3.5-turbo"  # fallback if your project lacks GPT-4 access
```

---

## ▶️ Running the App

```bash
streamlit run streamlit_app.py
```
Visit the URL shown (typically http://localhost:8501).

---
### 🧪 Using the App
## Single-Image Mode

1. **Upload Image(s)** – e.g., drone photo of a solar panel  
2. **Add Technician Notes** – e.g., “Crack on upper-left corner”  
3. **BLIP generates caption** – “A solar panel with visible cracks on surface.”  
4. **Merged context sent to GPT-4**  
5. **Generated Report Example**:

```json
{
  "Issue": "Cracked solar panel surface",
  "Component": "Solar panel module",
  "Severity": "High",
  "Recommendation": "Replace panel immediately to avoid energy loss",
  "Timeline": "Within 1 week"
}
```

---

## 📦 Batch Mode

- Upload a **ZIP file** containing:
  - Multiple solar panel images  
  - A `technician_notes.csv` file with two columns: `image_name, note`  

# 📦 ZIP Layout Example
Correct (files at any depth; script searches recursively):
```
my_batch.zip
├── data/
│   └── technician_notes.csv
└── images/
    ├── solar_panel1.JPG
    ├── solar_panel2.jpg
    └── solar_panel3.jpg
```
# CSV Format (required)

File name: technician_notes.csv
```
image_name,technical_note
solar_panel1.JPG,"Crack detected along the bottom-right cell; light burn marks."
solar_panel2.jpg,"Excessive dust on upper modules; inverter logging voltage dips."
solar_panel3.jpg,"Bird droppings causing shadows; reduced output confirmed."
```


The app will generate inspection reports for all images and allow you to **download results**.

---
### 🧩 Architecture (high-level)
```
[Image + Note Input]
        │
        ├─► BLIP (image → caption)
        │
        └─► LangChain Prompt (caption + note)
                     │
                     └─► OpenAI Chat Completions (GPT-4)
                                   │
                                   └─► JSON Report (Issue, Component, Severity, Recommendation, Timeline)
```

## 📘 Tech Stack

- [Streamlit](https://streamlit.io/) – frontend UI  
- [Transformers (BLIP)](https://huggingface.co/Salesforce/blip-image-captioning-base) – image captioning  
- [LangChain](https://www.langchain.com/) – prompt engineering  
- [OpenAI GPT-4](https://platform.openai.com/) – structured report generation  

---
## 🔒 Security Notes
- Don’t commit .env (already ignored by .gitignore)
- Prefer environment variables in production
- Consider rate limits & cost controls on your OpenAI project

## 📄 License
MIT License – feel free to use and modify for research, workshops, or enterprise solutions.

---

## 👨‍💻 Author
Developed by **Prakalp Somawanshi**  
Technologist & AI Leader | HPC | Energy Tech | GenAI  

---

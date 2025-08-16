
# Multimodal GenAI Solar Panel Inspection

This Streamlit app performs automated inspection of solar panels using drone images and technician notes. It uses:

- 🧠 BLIP for image captioning
- 🔤 LangChain for prompt engineering
- 💬 GPT-4 via OpenAI API for report generation
- 🎨 Streamlit for the interactive front-end

## 🔧 Setup Instructions

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:

- Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

4. Run the app:

```bash
streamlit run streamlit_app.py
```


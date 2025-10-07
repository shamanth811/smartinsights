# 🧠 SmartInsights — AI-Powered Data Storytelling App

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/shamanth111/smartinsights)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ✨ Turn raw CSV data into natural-language insights, smart visuals, and interactive Q&A — all in one click.

---

## 🚀 Live Demo
🟢 **Try it here:**  
👉 [**SmartInsights on Hugging Face Spaces**](https://huggingface.co/spaces/shamanth111/smartinsights)

---

## 📖 Overview
**SmartInsights** is a Streamlit-based AI companion that helps anyone understand their data faster.  
Upload your CSV/Excel file — the app analyzes, visualizes, and explains key trends in plain English using Hugging Face LLMs.

---

## ⚙️ Key Features

| Feature | Description |
|----------|-------------|
| 🧩 **Data Upload & Preview** | Upload CSV/Excel/JSON files or use demo data |
| 💬 **AI Summary Generator** | Automatic narrative summary of patterns, outliers, and trends |
| 🤖 **Conversational Insights** | Chat interface to ask questions about your dataset |
| 📈 **Smart Visuals Page** | Generates relevant graphs on demand |
| ⬇️ **Download Summary** | Export generated text for reports or notes |

---

## 🧠 Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | [Streamlit](https://streamlit.io) |
| Data | [Pandas](https://pandas.pydata.org), [Matplotlib](https://matplotlib.org), [Seaborn](https://seaborn.pydata.org) |
| LLM Engine | [Hugging Face Hub](https://huggingface.co) (Zephyr / Mistral models) |
| Orchestration | [LangChain](https://www.langchain.com) |
| Hosting | [Hugging Face Spaces](https://huggingface.co/spaces) |

---

## 🧭 How It Works
1. **Upload Data** → CSV/Excel/JSON accepted  
2. **Automatic Summary** → App runs statistical + text-based analysis  
3. **Generate Smart Visuals** → Optional charts built using Matplotlib/Seaborn  
4. **Chat with Data** → Ask questions in natural language  
5. **Download AI Summary** → One-click export  

---

## 📸 Screenshots

| Overview | Smart Visuals |
|-----------|----------------|
| ![Overview](screenshots/overview.png) | ![Visuals](screenshots/visuals.png) |

---

## 🧩 Local Setup

```bash
git clone https://github.com/shamanth811/smartinsights.git
cd smartinsights
python -m venv venv
venv\Scripts\activate  # (Windows)
pip install -r requirements.txt
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🪪 License
Released under the [MIT License](LICENSE).

---

## 💡 Author
**Shamanth** — [GitHub](https://github.com/shamanth811)  
🧩 Built as part of a Generative AI + Data Science portfolio series.

---

## ⭐ Support
If you like this project, please ⭐ the repo — it helps others discover it!

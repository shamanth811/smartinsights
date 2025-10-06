# app.py
import os
import io
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from huggingface_hub import InferenceClient

# ------------------------------
# 🎨 Page Setup
# ------------------------------
st.set_page_config(page_title="SmartInsights — Conversational Data AI", layout="wide")
st.title("🧠 SmartInsights — Your Conversational Data Companion")
st.caption("Upload your dataset, generate AI-powered summaries, visualize trends, and chat about your data.")

# ------------------------------
# ⚙️ Helper Functions
# ------------------------------
def load_dataset(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx"):
        return pd.read_excel(file)
    elif name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    elif name.endswith(".json"):
        return pd.read_json(file)
    else:
        st.error("Unsupported file type. Please upload CSV, Excel, TSV, or JSON.")
        st.stop()

def clean_data(df):
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols):
        imputer = SimpleImputer(strategy="median")
        df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

def generate_stats_summary(df):
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns
    stats = []
    for col in num_cols:
        stats.append(f"{col}: mean={df[col].mean():.2f}, min={df[col].min()}, max={df[col].max()}")
    for col in cat_cols:
        top_vals = df[col].value_counts().nlargest(3).to_dict()
        stats.append(f"{col}: top_values={top_vals}")
    return "\n".join(stats)

# ------------------------------
# 🧠 Hugging Face Setup
# ------------------------------
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("❌ Hugging Face API token not found.\nSet it using:\n`setx HUGGINGFACEHUB_API_TOKEN \"hf_your_token_here\"` and restart VS Code.")
    st.stop()

client = InferenceClient(model="HuggingFaceH4/zephyr-7b-alpha", token=HF_TOKEN)

def ask_model(prompt):
    """Ask the Hugging Face model a question."""
    try:
        response = client.chat_completion(
            model="HuggingFaceH4/zephyr-7b-alpha",
            messages=[
                {"role": "system", "content": "You are a clear, friendly data storytelling assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.5,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ Error: {e}"

# ------------------------------
# 📂 Upload Section
# ------------------------------
st.sidebar.header("📁 Upload or Use Demo")
uploaded_file = st.sidebar.file_uploader("Upload a dataset", type=["csv", "xlsx", "tsv", "json"])
use_demo = st.sidebar.checkbox("Use demo dataset (penguins.csv)", value=True if uploaded_file is None else False)

if uploaded_file:
    df = load_dataset(uploaded_file)
elif use_demo:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "penguins.csv"))
else:
    st.info("👈 Upload a dataset or enable demo mode to start.")
    st.stop()

df = clean_data(df)

# ------------------------------
# 🧭 Tabs Layout
# ------------------------------
tab1, tab2, tab3 = st.tabs(["📋 Overview", "📊 Smart Visuals", "❓ Help / FAQ"])

# ------------------------------
# 📋 Tab 1: Overview (Summary + Chat)
# ------------------------------
with tab1:
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown(f"**Rows:** {df.shape[0]} **Columns:** {df.shape[1]}")

    # Summary generation
    if st.button("🧠 Generate Summary"):
        with st.spinner("✨ Analyzing and summarizing your dataset..."):
            stats_text = generate_stats_summary(df)
            summary_prompt = f"""
Summarize this dataset in clear, beginner-friendly language.
Describe what it likely represents, key patterns, and one question to explore.

Dataset details:
{stats_text}
"""
            st.session_state.summary = ask_model(summary_prompt)

    if "summary" in st.session_state:
        st.subheader("🧠 AI Summary")
        st.write(st.session_state.summary)

        # Download summary
        summary_bytes = io.BytesIO(st.session_state.summary.encode("utf-8"))
        st.download_button(
            label="📥 Download Summary as Text",
            data=summary_bytes,
            file_name="smartinsights_summary.txt",
            mime="text/plain",
        )

        # Chat below summary
        st.divider()
        st.subheader("💬 Ask Questions About Your Dataset")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(content)

        if prompt := st.chat_input("Ask something about this dataset..."):
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            context = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\nColumns: {', '.join(df.columns[:10])}"
            chat_prompt = f"{context}\n\nUser question: {prompt}\nAnswer clearly, referencing summary insights where useful."
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = ask_model(chat_prompt)
                    st.markdown(response)
            st.session_state.chat_history.append(("assistant", response))
    else:
        st.info("👆 Click **Generate Summary** above to get started.")

# ------------------------------
# 📊 Tab 2: Smart Visuals
# ------------------------------
with tab2:
    st.subheader("📊 Smart Visualizations")
    st.caption("Generate correlation and category visuals — only if you approve.")

    generate = st.button("✅ Generate Visuals")

    if generate:
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include="object").columns

        if len(num_cols) >= 2:
            st.markdown("**Correlation Heatmap:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

        if len(cat_cols):
            st.markdown("**Category Distributions:**")
            for col in cat_cols[:2]:
                fig, ax = plt.subplots(figsize=(6, 3))
                df[col].value_counts().nlargest(5).plot(kind="bar", ax=ax, color="#4E79A7")
                ax.set_title(f"Top {col} Categories")
                st.pyplot(fig)
    else:
        st.info("Click **Generate Visuals** to create graphs.")

# ------------------------------
# ❓ Tab 3: Help / FAQ
# ------------------------------
with tab3:
    st.subheader("❓ Frequently Asked Questions")

    with st.expander("What file types can I upload?"):
        st.write("You can upload `.csv`, `.xlsx`, `.tsv`, or `.json` files.")

    with st.expander("When should I use Smart Visuals?"):
        st.write("Visuals are great for numeric or categorical data. For text-heavy or unstructured data, rely on the AI summary instead.")

    with st.expander("Can SmartInsights handle large files?"):
        st.write("It works best with datasets under 50 MB. For larger datasets, sample a few thousand rows.")

    with st.expander("How does the AI generate summaries?"):
        st.write("It analyzes basic statistics and top values, then uses Hugging Face models to generate clear, human-readable explanations.")

    with st.expander("Is my data private?"):
        st.write("Yes. Everything runs locally or within your Hugging Face Space — no external data storage.")

    with st.expander("Example questions you can ask:"):
        st.markdown("""
        - "Which column has the highest average value?"  
        - "What does this dataset seem to represent?"  
        - "What patterns do you notice between A and B?"  
        - "Are there any outliers?"  
        - "Which species has the longest flippers?"  
        """)

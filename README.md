# PaperLens-AI

### AI-Powered Research Paper Recommendation & Analysis System

Discover relevant research papers using semantic search, generate concise summaries, extract keywords, and analyze uploaded research papers with AI.

## ✨ Features

- 🔍 Semantic Research Paper Search
- 🤖 AI-based Research Paper Profiling
- 📚 Transformer-based Paper Summarization
- 🏷️ Automatic Keyword Extraction
- 📄 Upload & Analyze Research Papers (PDF)
- 🎯 Similarity Score using FAISS
- 💻 Interactive Streamlit Dashboard
- ⚡ Fast Semantic Retrieval

- ## 🛠 Tech Stack

- Python
- Streamlit
- Sentence Transformers
- FAISS
- HuggingFace Transformers
- BART
- KeyBERT
- Groq LLM (Llama 3.3)
- Pandas
- NumPy
- PyMuPDF

#Project Structure
- PaperLens/
│
├── app.py
├── recommendation.py
├── recommendation_fallback.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── paper_embeddings.npy
├── paper_faiss.index
│
├── profile_cache.json
│
└── assets/

#Workflow
User Query / PDF
        │
        ▼
Sentence Transformer
        │
        ▼
FAISS Semantic Search
        │
        ▼
Top Relevant Papers
        │
        ├──► BART Summarizer
        ├──► KeyBERT
        └──► Groq LLM
                    │
                    ▼
      Structured AI Insights
                    │
                    ▼
        Streamlit Dashboard

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/PaperLens.git
```

Move into the project folder

```bash
cd PaperLens
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**Mac/Linux**

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Run the Application

Start the Streamlit app

```bash
streamlit run app.py
```

Open your browser and visit

```
http://localhost:8501
```

## 📂 Dataset

- ML-ArXiv Papers Dataset
- Approximately 15,000 research papers

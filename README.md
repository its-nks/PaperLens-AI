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

## 📂 Project Structure

```text
PaperLens/
│
├── app.py                         # Streamlit application
├── recommendation.py              # Semantic search and AI pipeline
├── recommendation_fallback.py     # Backup recommendation module
├── requirements.txt               # Project dependencies
├── README.md
├── LICENSE
│
├── paper_embeddings.npy           # Precomputed sentence embeddings
├── paper_faiss.index              # FAISS vector index
├── profile_cache.json             # Cached AI-generated paper profiles
│
└── assets/                        # Screenshots and demo images
```

## ⚙️ Workflow

```text
                User Query / PDF
                        │
                        ▼
         Sentence Transformer Embedding
                        │
                        ▼
            FAISS Semantic Search
                        │
                        ▼
          Top Relevant Research Papers
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
 BART Summarizer   KeyBERT        Groq LLM
                        │
                        ▼
      Structured Research Paper Insights
                        │
                        ▼
          Interactive Streamlit Dashboard
```

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

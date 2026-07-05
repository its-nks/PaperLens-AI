import json
import os
import numpy as np
import pandas as pd
import faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT
from groq import Groq


# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading dataset...")

dataset = load_dataset(
    "CShorten/ML-ArXiv-Papers",
    split="train"
)

df = pd.DataFrame(dataset)

df = df[["title", "abstract"]]

df = df.head(15000)

df["paper_text"] = (
    df["title"] + " " + df["abstract"]
)

df["paper_text"] = (
    df["paper_text"]
    .str.replace("\n", " ", regex=False)
    .str.strip()
)

print("Dataset Loaded")
print(f"Total Papers : {len(df)}")


# ==========================================================
# SENTENCE TRANSFORMER
# ==========================================================

print("Loading SentenceTransformer...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("SentenceTransformer Ready")


# ==========================================================
# LOAD EMBEDDINGS
# ==========================================================

print("Loading embeddings...")

embeddings = np.load("paper_embeddings.npy")

print("Embeddings Loaded")


# ==========================================================
# LOAD FAISS INDEX
# ==========================================================

print("Loading FAISS Index...")

index = faiss.read_index(
    "paper_faiss.index"
)

print("FAISS Ready")


# ==========================================================
# LOAD SUMMARIZER
# ==========================================================

print("Loading BART Summarizer...")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

print("Summarizer Ready")


# ==========================================================
# LOAD KEYBERT
# ==========================================================

print("Loading KeyBERT...")

kw_model = KeyBERT(
    embedding_model
)

print("KeyBERT Ready")


# ==========================================================
# GROQ CLIENT
# ==========================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ==========================================================
# AI RESEARCH PROFILE
# ==========================================================

def generate_paper_profile(title, abstract):

    prompt = f"""
You are an expert AI Research Assistant.

Analyze the following research paper.

Return ONLY valid JSON.

Rules:
1. Return exactly one value for each field.
2. If information is unavailable, write "Not Mentioned".
3. Contribution must be exactly one sentence.
4. Difficulty must be one of:
   Beginner
   Intermediate
   Advanced

Return ONLY:

{{
    "Research Area":"",
    "Method":"",
    "Dataset":"",
    "Task":"",
    "Contribution":"",
    "Difficulty":"",
    "Best For":"",
    "Estimated Reading Time":""
}}

Title:
{title}

Abstract:
{abstract}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    text = completion.choices[0].message.content.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    try:
        profile = json.loads(text)

    except json.JSONDecodeError:

        profile = {
            "Research Area": "Not Mentioned",
            "Method": "Not Mentioned",
            "Dataset": "Not Mentioned",
            "Task": "Not Mentioned",
            "Contribution": text,
            "Difficulty": "Intermediate",
            "Best For": "Researchers",
            "Estimated Reading Time": "1 minute"
        }

    return profile

# ==========================================================
# SEARCH + SUMMARIZE + AI INSIGHTS
# ==========================================================

def search_and_summarize(query, k=5):

    # Encode query
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)

    # Search similar papers
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        title = df.iloc[idx]["title"]
        abstract = df.iloc[idx]["abstract"]

        # -----------------------------
        # Generate Summary
        # -----------------------------

        summary = summarizer(
            abstract,
            max_length=120,
            min_length=40,
            do_sample=False
        )[0]["summary_text"]

        # -----------------------------
        # Extract Keywords
        # -----------------------------

        keywords = kw_model.extract_keywords(
            abstract,
            top_n=5,
            stop_words="english"
        )

        keyword_list = [word for word, _ in keywords]

        # -----------------------------
        # AI Profile
        # -----------------------------

        profile = generate_paper_profile(
            title,
            abstract
        )

        # -----------------------------
        # Store Everything
        # -----------------------------

        results.append({

            "match_score": round(float(score) * 100, 2),

            "title": title,

            "abstract": abstract,

            "summary": summary,

            "keywords": keyword_list,

            "profile": profile

        })

    return results

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    papers = search_and_summarize(
        "Deep Learning in Medical Imaging",
        k=2
    )

    print(papers[0]["title"])

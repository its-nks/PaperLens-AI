import fitz
import re
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "will", "with", "using", "study",
    "paper", "research", "based", "model", "models", "method", "methods"
}

DEFAULT_PAPERS: List[Dict[str, str]] = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The Transformer architecture relies entirely on self-attention to model global dependencies in sequence data and has become a foundation for modern NLP systems.",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "BERT introduces bidirectional pretraining for language representations and achieves strong performance across a wide range of natural language understanding tasks.",
    },
    {
        "title": "Deep Residual Learning for Image Recognition",
        "abstract": "Residual networks make it possible to train very deep convolutional neural networks by introducing shortcut connections that improve optimization and accuracy.",
    },
    {
        "title": "Generative Adversarial Networks",
        "abstract": "GANs train a generator and discriminator adversarially to produce realistic synthetic data and have influenced modern generative modeling.",
    },
    {
        "title": "A Survey of Deep Learning for Medical Imaging",
        "abstract": "This survey reviews deep learning methods applied to medical imaging, highlighting common architectures, datasets, and clinical challenges in the field.",
    },
    {
        "title": "Contrastive Learning of Medical Visual Representations from Paired Images and Text",
        "abstract": "This work uses contrastive learning to align medical images and text, improving representation learning for diagnostic and clinical applications.",
    },
    {
        "title": "Graph Neural Networks for Recommendation Systems",
        "abstract": "This paper describes how graph neural networks can model user-item interactions to improve recommendation quality and personalization.",
    },
    {
        "title": "Exploring Reinforcement Learning in Robotics",
        "abstract": "The study investigates deep reinforcement learning methods for robotic control tasks and evaluates sample efficiency across environments.",
    },
    {
        "title": "Self-Supervised Learning for Computer Vision",
        "abstract": "Self-supervised representation learning is used to pretrain vision models without labeled data and improves performance on downstream tasks.",
    },
    {
        "title": "Audio-Visual Speech Recognition with Transformers",
        "abstract": "A multimodal transformer combines audio and video streams to enhance speech recognition robustness under noisy conditions.",
    },
    {
        "title": "Efficient Neural Architecture Search for Embedded Systems",
        "abstract": "This paper presents an efficient approach to neural architecture search (NAS) tailored for resource-constrained embedded devices.",
    },
    {
        "title": "Federated Learning for Privacy-Preserving Healthcare Analytics",
        "abstract": "Federated learning enables collaborative model training on medical data while keeping patient records private and decentralized.",
    },
    {
        "title": "Zero-Shot Learning with Vision-Language Models",
        "abstract": "A zero-shot learning method combines vision-language representations to classify unseen categories without task-specific fine-tuning.",
    },
    {
        "title": "Anomaly Detection in Time Series Data",
        "abstract": "This work studies anomaly detection algorithms for time series, including deep autoencoders and contrastive learning techniques.",
    },
    {
        "title": "Personalized Recommendation with Hybrid Collaborative Filtering",
        "abstract": "A hybrid collaborative filtering system combines matrix factorization and content-based features to improve personalization accuracy.",
    },
]

CORPUS = [f"{paper['title']} {paper['abstract']}" for paper in DEFAULT_PAPERS]
VECTORIZER = TfidfVectorizer(stop_words="english")
DOCUMENT_VECTORS = VECTORIZER.fit_transform(CORPUS)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", _clean_text(text).lower())
    return [token for token in tokens if token not in STOP_WORDS]


def _summarize_abstract(abstract: str) -> str:
    cleaned = _clean_text(abstract)
    if not cleaned:
        return "No abstract available."
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    first_sentence = next((s for s in sentences if s), cleaned)
    if len(first_sentence) <= 220:
        return first_sentence
    return first_sentence[:217] + "..."


def _extract_keywords(text: str, top_n: int = 5) -> List[str]:
    freq: Dict[str, int] = {}
    for token in _tokenize(text):
        freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:top_n]]


def generate_paper_profile(title: str, abstract: str) -> Dict[str, str]:
    title_lower = title.lower()
    abstract_lower = abstract.lower()

    if "medical" in abstract_lower or "clinical" in abstract_lower:
        research_area = "Medical AI"
        difficulty = "Intermediate"
    elif "transformer" in title_lower or "transformer" in abstract_lower:
        research_area = "Natural Language Processing"
        difficulty = "Advanced"
    elif "vision" in abstract_lower or "image" in abstract_lower:
        research_area = "Computer Vision"
        difficulty = "Intermediate"
    else:
        research_area = "Machine Learning"
        difficulty = "Beginner"

    if "image" in abstract_lower or "vision" in abstract_lower:
        method = "Computer Vision"
    elif "transformer" in abstract_lower or "bert" in title_lower:
        method = "Transformer-based Model"
    elif "graph" in abstract_lower:
        method = "Graph Neural Networks"
    elif "federated" in abstract_lower or "privacy" in abstract_lower:
        method = "Federated Learning"
    else:
        method = "Deep Learning"

    if "medical" in abstract_lower:
        dataset = "Medical imaging or clinical data"
    elif "text" in abstract_lower or "speech" in abstract_lower:
        dataset = "Text or audio corpus"
    elif "graph" in abstract_lower:
        dataset = "Graph-structured data"
    else:
        dataset = "Benchmark dataset"

    if "classification" in abstract_lower or "recognition" in abstract_lower:
        task = "Prediction or classification"
    elif "generate" in abstract_lower or "generative" in abstract_lower:
        task = "Generation"
    elif "anomaly" in abstract_lower:
        task = "Anomaly detection"
    else:
        task = "Representation learning"

    contribution = "This paper introduces a practical method that improves performance or understanding in its target domain."

    return {
        "Research Area": research_area,
        "Method": method,
        "Dataset": dataset,
        "Task": task,
        "Contribution": contribution,
        "Difficulty": difficulty,
        "Best For": "Researchers and students",
        "Estimated Reading Time": "5-8 minutes",
    }


def _text_similarity(query: str) -> List[float]:
    query = _clean_text(query)
    if not query:
        return [0.0] * len(CORPUS)
    query_vector = VECTORIZER.transform([query])
    similarities = cosine_similarity(query_vector, DOCUMENT_VECTORS).flatten()
    return similarities.tolist()


def search_and_summarize(query: str, k: int = 5) -> List[Dict[str, object]]:
    if not query or not str(query).strip():
        return []

    similarities = _text_similarity(query)
    scored_results = []

    for idx, paper in enumerate(DEFAULT_PAPERS):
        title = paper["title"]
        abstract = paper["abstract"]
        score = similarities[idx]
        title_bonus = 0.1 if any(term in title.lower() for term in _tokenize(query)) else 0.0
        score += title_bonus

        scored_results.append({
            "match_score": round(min(score * 100.0, 100.0), 2),
            "title": title,
            "abstract": abstract,
            "summary": _summarize_abstract(abstract),
            "keywords": _extract_keywords(title + " " + abstract),
            "profile": generate_paper_profile(title, abstract),
        })

    scored_results.sort(key=lambda item: item["match_score"], reverse=True)
    return scored_results[: max(1, int(k))]


def analyze_pdf(uploaded_file) -> Dict[str, object]:
    raw_bytes = uploaded_file.read()
    document = fitz.open(stream=raw_bytes, filetype="pdf")
    text = []
    for page in document:
        page_text = _clean_text(page.get_text())
        if page_text:
            text.append(page_text)
    full_text = "\n\n".join(text)
    document.close()

    summary = _summarize_abstract(full_text)
    keywords = _extract_keywords(full_text)
    title = getattr(uploaded_file, "name", "Uploaded Paper")
    profile = generate_paper_profile(title, full_text)

    return {
        "title": title,
        "summary": summary,
        "keywords": keywords,
        "profile": profile,
        "text": full_text,
    }


if __name__ == "__main__":
    papers = search_and_summarize("Deep Learning in Medical Imaging", k=2)
    print(papers[0]["title"])

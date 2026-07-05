import streamlit as st

try:
    from recommendation_fallback import search_and_summarize, analyze_pdf
except Exception as exc:
    search_and_summarize = None
    analyze_pdf = None
    import_error = exc
else:
    import_error = None

st.set_page_config(
    page_title="Research Paper Recommendation",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero-card {
        background: transparent;
        border: none;
        padding: 1.5rem 0 1rem;
        box-shadow: none;
        margin-bottom: 1.2rem;
    }
    .section-card {
        background: transparent;
        border: none;
        padding: 0;
        box-shadow: none;
        margin-bottom: 0.8rem;
    }
    .result-card {
        background: transparent;
        border: none;
        padding: 0.6rem 0;
        margin-bottom: 0.85rem;
    }
    .keyword-pill {
        display: inline-block;
        margin: 0 0.35rem 0.35rem 0;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: #dde7f5;
        color: #1f3254;
        font-size: 0.9rem;
    }
    .metric-label {
        color: #334155;
        font-size: 0.92rem;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
    }
    .section-title {
        margin-bottom: 0.6rem;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.subheader("Research paper recommendation")
    st.title("Find the most relevant papers fast")
    st.markdown(
        "<p style='color:#475569; font-size:1rem; line-height:1.75; max-width:760px;'>Use a simple topic query to discover the best matching papers, or upload a PDF to extract insights directly from a document.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

option = st.radio(
    "Mode",
    ["Search by Topic", "Upload PDF"],
    horizontal=True,
)

if option == "Search by Topic":
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Search research papers")
        st.write("Enter a query and review the top-ranked papers with concise summaries and keyword highlights.")

        query_col, count_col = st.columns([4, 1])
        with query_col:
            query = st.text_input(
                "Query",
                placeholder="e.g. deep learning for medical imaging",
                key="query_input",
            )
        with count_col:
            top_k = st.number_input(
                "Results",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key="top_k",
            )

        if st.button("Search papers", use_container_width=True):
            if not query.strip():
                st.warning("Please enter a search query.")
            elif search_and_summarize is None:
                st.error(f"Search engine unavailable: {import_error}")
            else:
                with st.spinner("Finding relevant papers..."):
                    st.session_state["search_results"] = search_and_summarize(query, top_k)

        st.markdown('</div>', unsafe_allow_html=True)

    results = st.session_state.get("search_results", [])
    if results:
        for idx, paper in enumerate(results, start=1):
            profile = paper["profile"]
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            row1, row2 = st.columns([4, 1])
            with row1:
                st.markdown(f"<strong>{idx}. {paper['title']}</strong>", unsafe_allow_html=True)
            with row2:
                st.markdown('<div class="metric-label">Match score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{paper["match_score"]}%</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Summary**")
            st.write(paper["summary"])

            st.markdown("**Keywords**")
            keyword_html = "".join([f"<span class='keyword-pill'>{keyword}</span>" for keyword in paper["keywords"]])
            st.markdown(keyword_html, unsafe_allow_html=True)

            st.markdown("**Insights**")
            insight_left, insight_right = st.columns(2)
            with insight_left:
                st.write(f"**Area:** {profile['Research Area']}")
                st.write(f"**Method:** {profile['Method']}")
                st.write(f"**Dataset:** {profile['Dataset']}")
            with insight_right:
                st.write(f"**Task:** {profile['Task']}")
                st.write(f"**Difficulty:** {profile['Difficulty']}")
                st.write(f"**Reading time:** {profile['Estimated Reading Time']}")

            st.markdown("**Contribution**")
            st.write(profile["Contribution"])

            with st.expander("View abstract"):
                st.write(paper["abstract"])
            st.markdown('</div>', unsafe_allow_html=True)
    elif "search_results" in st.session_state:
        st.info("No papers matched that query. Try a broader search.")

else:
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Upload PDF for analysis")
        st.write("Upload a PDF and receive a structured summary, keywords, and paper insights.")

        uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

        if uploaded_file:
            if analyze_pdf is None:
                st.error(f"PDF analysis unavailable: {import_error}")
            else:
                with st.spinner("Analyzing uploaded document..."):
                    analysis = analyze_pdf(uploaded_file)

                st.success("PDF analysis complete.")
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"<strong>{analysis['title']}</strong>", unsafe_allow_html=True)
                st.markdown("---")

                st.markdown("**Summary**")
                st.write(analysis["summary"])

                st.markdown("**Keywords**")
                keyword_html = "".join([f"<span class='keyword-pill'>{keyword}</span>" for keyword in analysis["keywords"]])
                st.markdown(keyword_html, unsafe_allow_html=True)

                st.markdown("**Insights**")
                insight_left, insight_right = st.columns(2)
                profile = analysis["profile"]
                with insight_left:
                    st.write(f"**Area:** {profile['Research Area']}")
                    st.write(f"**Method:** {profile['Method']}")
                    st.write(f"**Dataset:** {profile['Dataset']}")
                with insight_right:
                    st.write(f"**Task:** {profile['Task']}")
                    st.write(f"**Difficulty:** {profile['Difficulty']}")
                    st.write(f"**Reading time:** {profile['Estimated Reading Time']}")

                st.markdown("**Contribution**")
                st.write(profile["Contribution"])

                st.markdown("**Extracted text preview**")
                preview = analysis["text"][:1800]
                st.write(preview + ("..." if len(analysis["text"]) > 1800 else ""))
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

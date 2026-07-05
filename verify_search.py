from recommendation_fallback import search_and_summarize
for query in ["medical imaging", "graph recommendation", "speech recognition", "transformer NLP"]:
    results = search_and_summarize(query, k=3)
    print('QUERY:', query)
    for r in results:
        print(r['match_score'], r['title'])
    print('---')

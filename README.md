HN Search based on profile description.

Optimized for easiness to code in 3 hours. Currently slow and suboptimal (see "Will Not Do" section).

Summary
- Extract stemmed keywords
- Query HN Algolia
- Apply CrossEncoder to (description, comments / titles) to get score

```
pip install aiohttp flask[async] transformers sentence-transformers nltk
python -m nltk.downloader stopwords punkt
python search.py
```

# Example Profile Description

I am a theoretical biologist, interested in disease ecology. My tools are R, clojure , compartmentalism disease modeling, and statistical GAM models, using a variety of data layers (geophysical, reconstructions, climate, biodiversity, land use). Besides that I am interested in tech applied to the a subset of the current problems of the world (agriculture / biodiversity / conservation / forecasting), development of third world countries and AI, large language models.

# Process

## Extract stemmed terms w/ stopword removal

HN search API finds literal strings, doesn't appear to be fuzzy at all, description has things like "compartmentalism disease modeling", while an article about a "compartmentalized disease model" would be relevant.

Quick and easy solution is stemming / stopword removal to get query keywords for Algolia API.

```
['theoret', 'biologist', 'interest', 'diseas', 'ecolog', 'tool', 'r', 'clojur', 'compartment', 'diseas', 'model', 'statist', 'gam', 'model', 'use', 'varieti', 'data', 'layer', 'geophys', 'reconstruct', 'climat', 'biodivers', 'land', 'use', 'besid', 'interest', 'tech', 'appli', 'subset', 'current', 'problem', 'world', 'agricultur', 'biodivers', 'conserv', 'forecast', 'develop', 'third', 'world', 'countri', 'ai', 'larg', 'languag', 'model']
```

## Retrieve and Re-Rank

In my experience cross-encoders are more accurate than bi-encoders. They work when you have a smaller corpus, but don't scale as well as ANN. I will use SentencePiece CrossEncoder for re-ranking. It'd be nice to have a bi-encoder for initial filtering, but out of scope for a 3 hour project.

## Display results

Return the ranked titles, comments, and respective scores. Serve via Flask.


# Will Not Do

Out of scope due to 3 hour time constraint, but would improve result quality:

- A variety of heuristics including recency, comment count, TF-IDF like weighing for portions of comments matched;
- Keyword generation model to make things like "R" more searchable (currently has irrelevant results for things like "r/place")
- Balancing so different keywords aren't overrepresented (currently the top results mostly pertain to Clojure)
- Rank with BM25 / bi-encoder before re-ranking with cross-encoder
- Guarantee there are 500 results by extending Algolia queries if there are an insufficient number of results
- Improve results by retrieving contents and/or embeddings of submissions linked web pages

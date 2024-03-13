import aiohttp
import asyncio
import collections
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sentence_transformers import CrossEncoder, util

import urllib.parse

from flask import Flask, request, jsonify, redirect, url_for
app = Flask(__name__)


cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2", max_length=256)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def get_keywords(description):
    # tokenize, filter stop words, filter non-alphanumeric, stem, quote unstemmed
    tokens = word_tokenize(description)
    tokens = [t for t in tokens if t.lower() not in stop_words]
    tokens = map(stemmer.stem, tokens)
    tokens = filter(str.isalnum, tokens)
    return list(tokens)


async def query_hn(session, query_str, tags, hits_per_page=50):
    params = {"query": query_str, "hitsPerPage": hits_per_page, "tags": tags}
    async with session.get("http://hn.algolia.com/api/v1/search", params=params) as response:
        return await response.json()


async def search_hn(description):
    keywords = get_keywords(description)

    async with aiohttp.ClientSession() as session:
        coroutines = [query_hn(session, kw, tags="comment") for kw in keywords]
        for coroutine in asyncio.as_completed(coroutines):
            response = await coroutine
            yield response["hits"]


async def get_best_submissions(desc, n=500):
    results = collections.defaultdict(dict)
    async for hits in search_hn(desc):
        comment_scores = cross_enc.predict([[desc, h['comment_text']] for h in hits])
        title_scores = cross_enc.predict([[desc, h['story_title']] for h in hits])
        for hit, comment_score, title_score in zip(hits, comment_scores, title_scores):
            story_dict = results[hit["story_id"]]
            story_dict["title"] = hit["story_title"]
            story_dict["title_score"] = float(title_score)
            story_dict["comments"] = story_dict.get("comments", {})
            story_dict["comments"][hit["objectID"]] = {
                "comment_score": float(comment_score),
                # hacky strip html and truncate
                "comment_text": re.sub('<[^<]+?>', '', hit["comment_text"])[:256] + "...",

            }

    for sid, res in results.items():
        comm_factor = 1 / (4 * len(res["comments"]))**0.5
        best_comm_score = max([c["comment_score"] for c in res["comments"].values()])
        results[sid]["score"] = (best_comm_score * comm_factor + res["title_score"]) / (1 + comm_factor)

    best_results = sorted(results.items(), key=lambda item: item[1]["score"], reverse=True)
    return best_results[:n]


@app.route('/', methods=['GET'])
async def index():
    # Serving a simple form
    return '''
    <html>
    <body>
        <form action="/search" method="get">
            <label for="desc">HN profile description:</label><br>
            <textarea id="desc" name="desc" rows="4" cols="50"></textarea><br>
            <input type="submit" value="Search">
        </form>
    </body>
    </html>
    '''

@app.route('/search/')
async def search_results():
    desc = request.args.get('desc')
    return """
    <html>
    <body>
        <p>Loading results...</p>
        <script>
        fetch('/best_submissions/?desc=""" + urllib.parse.quote(desc) + """')
        .then(response => response.json())
        .then(data => {
             const submissionsHTML = data.map(([id, {score, title, title_score, comments}]) => {
                 const sortedComments = Object.entries(comments).sort(([,a], [,b]) => b.comment_score - a.comment_score).slice(0, 3);
                 return `
                     <li>
                        <a href="https://news.ycombinator.com/item?id=${id}">${title}</a><br>
                        <small><i>(Overall Score: ${score.toFixed(2)})</i></small>
                        <small><i>(Title Score: ${title_score.toFixed(2)})</i></small><br>
                        <ul>
                        ${sortedComments.map(([comment_id, {comment_score, comment_text}]) => `
                            <li>
                            <a href="https://news.ycombinator.com/item?id=${comment_id}">${comment_text}</a>
                            <small><i>(Comment Score: ${comment_score.toFixed(2)})</i></small>
                            </li>
                        `).join('')}
                        </ul>
                    </li>
                `;
             }).join('<hr/>');
            document.body.innerHTML = `<ul>${submissionsHTML}</ul>`;
            });
        </script>
    </body>
    </html>
    """


@app.route('/best_submissions/')
async def best_submissions():
    desc = request.args.get('desc')
    submissions = await get_best_submissions(desc.strip())
    return jsonify(submissions)


if __name__ == '__main__':
    app.run(debug=True)

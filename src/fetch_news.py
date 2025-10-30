"""
fetch_news.py fetches news articles about "AI" from the NewsAPI 'https://newsapi.org/'
"""

# imports
import os
import requests
from dotenv import load_dotenv

def fetch_news(query="AI", page_size=100, language="en"):
    """
    fetches recent news articles related to the given query (default: "AI")
    using the NewsAPI.

    :param query (str): the query to search for
    :param page_size (int): the number of results (max 100)
    :param language (str): the language of the news articles

    :return: list[dict]
    """

    # load API key from .env file
    load_dotenv()
    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        raise ValueError("API key not found. Please add NEWS_API_KEY to your .env file.")

    # NewsAPI endpoint
    url = "https://newsapi.org/v2/everything"

    # example API request: GET https://newsapi.org/v2/everything?q=AI&from=2025-10-30&sortBy=popularity&apiKey=API_KEY

    params = {
        "q": query,
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key
    }

    print(f"Fetching latest news about '{query}' ...")

    response = requests.get(url, params=params)

    # Handle errors
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    data = response.json()

    if "articles" not in data:
        raise ValueError("Articles not found - check the API response structure.")

    articles = data["articles"]

    # extract only the needed fields
    extracted_articles = []
    for article in articles:
        extracted_articles.append({
            "source": article["source"]["name"] if article["source"] else None,
            "author": article.get("author"),
            "title": article.get("title"),
            "description": article.get("description"),
            "content": article.get("content"),
            "url": article.get("url"),
            "publishedAt": article.get("publishedAt"),
        })

    print(f"Fetched {len(extracted_articles)} articles! ₍^. .^₎⟆")

    return extracted_articles

if __name__ == "__main__":
    try:
        articles = fetch_news(query="AI", page_size=100)
    except Exception as e:
        print(f"Something went wrong ૮₍•᷄  ༝ •᷅₎ა --> Error message: {type(e).__name__} - {e}.")

"""
fetch_news.py fetches news articles about "AI" from the NewsAPI 'https://newsapi.org/'
"""

# imports
import os
import requests
from dotenv import load_dotenv

class NewsFetcher:
    """a class for fetching news articles from the NewsAPI"""

    def __init__(self, query="AI", page_size=100, language="en"):
        """
        initialize the NewsFetcher with search parameters.

        :param query: The search keyword (default: "AI").
        :param page_size: The number of results to fetch (max 100).
        :param language: The language of the news articles.
        """
        self.query = query
        self.page_size = page_size
        self.language = language
        self.api_key = None
        self.url = "https://newsapi.org/v2/everything"

    def load_api_key(self):
        """loads the NewsAPI key from a .env file."""
        load_dotenv()
        self.api_key = os.getenv("NEWS_API_KEY")

        if not self.api_key:
            raise ValueError("API key not found. Please add NEWS_API_KEY to your .env file /ᐠ - ˕ -マ.")

    def fetch(self):
        """
        fetches recent news articles from the NewsAPI.

        :return: List of dictionaries containing article data.
        """
        if not self.api_key:
            self.load_api_key()

        params = {
            "q": self.query,
            "language": self.language,
            "sortBy": "publishedAt",
            "pageSize": self.page_size,
            "apiKey": self.api_key,
        }

        print(f"... ₍^. .^₎⟆ Fetching latest news about '{self.query}' ...")

        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"/ᐠ•˕•マ? Couldn't fetch the requested API : {e}.")
            return []

        data = response.json()

        if "articles" not in data:
            raise ValueError("Articles not found — check the API response structure ≽^- ˕ -^≼")

        articles = data["articles"]

        # Extract only relevant fields
        cleaned_articles = [
            {
                "source": article["source"]["name"] if article["source"] else None,
                "author": article.get("author"),
                "title": article.get("title"),
                "description": article.get("description"),
                "content": article.get("content"),
                "url": article.get("url"),
                "publishedAt": article.get("publishedAt"),
            }
            for article in articles
        ]

        print(f"Fetched {len(cleaned_articles)} articles successfully ദ്ദി/ᐠ｡‸｡ᐟ\!")
        return cleaned_articles


if __name__ == "__main__":
    try:
        fetcher = NewsFetcher(query="AI", page_size=100)
        articles = fetcher.fetch()

        if not articles:
            print("No articles found ... ฅ^•ﻌ•^ฅ ...")

        else:
            print("Examples of the first five articles:\n")
            for idx, article in enumerate(articles[:5], start=1):
                source = article.get("source") or "Unknown source"
                title = article.get("title") or "N/a"
                url = article.get("url") or "N/a"
                print(f"Article {idx} | Title: {title}.")
                print(f"- Source: {source}")
                print(f"- Url: {url}\n")

    except Exception as e:
        print(f"Couldn't fetch the requested articles... ^. .^₎Ⳋ Error type: {type(e).__name__} | Error message: {e}.")

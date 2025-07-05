# newsapi_utils.py

import requests

def fetch_top_headlines(api_key, category='technology', limit=5):
    if not api_key:
        print("❌ API key is missing.")
        return ["❌ API key is missing. Please set it in your .env file."]

    url = f"https://newsapi.org/v2/top-headlines?category={category}&pageSize={limit}&apiKey={api_key}"

    try:
        response = requests.get(url, timeout=5)
        print("🔗 Request URL:", url)
        print("🔁 Status Code:", response.status_code)

        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            if not articles:
                print("📭 No headlines found.")
                return ["📭 No headlines returned by NewsAPI."]
            headlines = [article.get("title", "No Title") for article in articles]
            return headlines

        elif response.status_code == 401:
            return ["🔐 Error 401: Unauthorized. Check your NewsAPI key."]

        elif response.status_code == 429:
            return ["⏱ Error 429: Too Many Requests. Rate limit exceeded."]

        else:
            return [f"⚠️ API error: {response.status_code} - {response.reason}"]

    except requests.exceptions.RequestException as e:
        return [f"❌ Network error: {e}"]

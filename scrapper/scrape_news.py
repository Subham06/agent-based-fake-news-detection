import requests
import pandas as pd
import time

API_KEY = "YOUR_NEWSAPI_KEY"
URL = "https://newsapi.org/v2/everything"

all_articles = []
page = 1

# NewsAPI allows pagination (max 100 per page)
while len(all_articles) < 1000:
    params = {
        "q": "india OR world OR technology OR business",
        "language": "en",
        "pageSize": 100,
        "page": page,
        "apiKey": API_KEY
    }

    response = requests.get(URL, params=params)
    data = response.json()

    if data["status"] != "ok":
        print("Error:", data)
        break

    articles = data.get("articles", [])
    
    if not articles:
        break

    for article in articles:
        all_articles.append({
            "source": article["source"]["name"],
            "author": article["author"],
            "title": article["title"],
            "description": article["description"],
            "url": article["url"],
            "publishedAt": article["publishedAt"],
            "content": article["content"]
        })

    print(f"Collected {len(all_articles)} articles...")
    page += 1

    # Avoid rate limit
    time.sleep(1)

# Trim to 1000
all_articles = all_articles[:1000]

# Save JSON
import json
with open("news_data.json", "w", encoding="utf-8") as f:
    json.dump(all_articles, f, indent=4)

# Save CSV
df = pd.DataFrame(all_articles)
df.to_csv("news_data.csv", index=False, encoding="utf-8")

print("Done! Saved news_data.json and news_data.csv")
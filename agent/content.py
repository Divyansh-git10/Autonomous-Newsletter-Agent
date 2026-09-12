from tools.article_extractor import extract_article_text


def content_enrichment_node(state: dict) -> dict:
    """
    Attempts to fetch full article content for each raw article, but
    ONLY for articles whose URL was actually verified as a direct
    publisher link. A Google News wrapper URL is never handed to the
    extractor as if it were the article itself -- that's an interstitial
    redirect page, not real content, and treating it as such risks
    scraping noise text into the newsletter.
    """

    raw_articles = state.get("raw_articles", [])

    enriched_articles: list[dict] = []
    counts = {
        "success": 0,
        "failed_http_error": 0,
        "failed_timeout": 0,
        "failed_blocked": 0,
        "failed_empty": 0,
        "skipped_wrapper_url": 0,
        "skipped_no_url": 0,
    }

    for index, article in enumerate(raw_articles, start=1):
        title = article.get("title", "")
        print(f"Extracting article {index}/{len(raw_articles)}: {title}")

        is_verified = (
            article.get("url_resolution_status") == "verified_direct"
            and bool(article.get("resolved_url"))
        )

        if is_verified:
            extraction = extract_article_text(article["resolved_url"])
        elif article.get("rss_url") or article.get("url"):
            print("Skipping extraction: no verified publisher URL (Google News reference only).")
            extraction = {
                "content": "",
                "content_available": False,
                "extraction_status": "skipped_wrapper_url",
                "source_url": article.get("rss_url") or article.get("url", ""),
            }
        else:
            print("Skipping extraction: no URL available at all.")
            extraction = {
                "content": "",
                "content_available": False,
                "extraction_status": "skipped_no_url",
                "source_url": "",
            }

        counts[extraction["extraction_status"]] = counts.get(extraction["extraction_status"], 0) + 1

        updated_article = dict(article)
        updated_article["content"] = extraction["content"]
        updated_article["content_available"] = extraction["content_available"]
        updated_article["extraction_status"] = extraction["extraction_status"]
        updated_article["is_title_only"] = not extraction["content_available"]

        enriched_articles.append(updated_article)

    print(
        "Content extraction summary: "
        f"{counts['success']} succeeded, "
        f"{counts['skipped_wrapper_url']} skipped (Google News reference only), "
        f"{counts['failed_empty']} empty/too short, "
        f"{counts['failed_http_error']} HTTP errors, "
        f"{counts['failed_timeout']} timeouts, "
        f"{counts['failed_blocked']} blocked pages, "
        f"{counts['skipped_no_url']} with no URL at all."
    )

    return {
        "raw_articles": enriched_articles,
    }


if __name__ == "__main__":
    test_state = {
        "raw_articles": [
            {
                "title": "Example article",
                "rss_url": "https://news.google.com/rss/articles/fake",
                "url": "https://news.google.com/rss/articles/fake",
                "url_resolution_status": "unresolved",
                "resolved_url": "",
                "source": "Example",
                "published": "",
                "summary": "This is fallback RSS summary.",
            }
        ]
    }

    result = content_enrichment_node(test_state)

    for article in result["raw_articles"]:
        print("\nTitle:", article["title"])
        print("Extraction status:", article["extraction_status"])
        print("Content available:", article["content_available"])

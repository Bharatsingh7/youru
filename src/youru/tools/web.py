"""Internet tools: search, news, read a page, weather. No API keys needed."""

from __future__ import annotations

import urllib.parse

import httpx

from . import tool

_CACHE: dict[tuple, tuple[float, list]] = {}


def _ddgs(kind: str, query: str, n: int) -> list[dict]:
    """DuckDuckGo first (~0.8 s), Bing if DuckDuckGo refuses (it rate-limits bursts).
    Identical searches within 10 minutes are answered from a cache."""
    import time

    from ddgs import DDGS

    key = (kind, query.lower().strip(), n)
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] < 600:
        return hit[1]
    search = getattr(DDGS(timeout=5), kind)
    results: list = []
    for backend in ("duckduckgo", "bing"):
        try:
            results = search(query, max_results=n, backend=backend)
        except Exception:  # noqa: BLE001  (ddgs raises when an engine refuses or finds nothing)
            continue
        if results:
            break
    _CACHE[key] = (time.monotonic(), results)
    return results


UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/141.0 Safari/537.36"}


@tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo) for current information. Use for facts you are not sure about,
    prices, people, sports scores, anything that may have changed recently.

    Args:
        query: what to search for
        max_results: number of results (1-10)
    """
    results = _ddgs("text", query, max(1, min(10, max_results)))
    if not results:
        return "no results"
    return "\n".join(f"{i}. {r['title']} - {r['body']} ({r['href']})" for i, r in enumerate(results, 1))


@tool()
def news_search(query: str, max_results: int = 5) -> str:
    """Search recent news articles. Write the query in English.

    Args:
        query: news topic in English, e.g. "ISRO", "India cricket" or "India" for top headlines
        max_results: number of articles (1-10)
    """
    n = max(1, min(10, max_results))
    results = _ddgs("news", query, n)
    if not results:  # e.g. a Hinglish query like "taaza khabren": fall back to top Indian headlines
        results = _ddgs("news", "India", n)
    if not results:
        return "no news found"
    return "\n".join(f"{i}. [{r.get('date', '')[:10]}] {r['title']} ({r.get('source', '')}) - {r['body']}"
                     for i, r in enumerate(results, 1))


@tool()
def fetch_page(url: str) -> str:
    """Download a web page and return its main text. Use after web_search to read a result in detail.

    Args:
        url: full URL of the page
    """
    import trafilatura

    r = httpx.get(url, headers=UA, timeout=12, follow_redirects=True)
    r.raise_for_status()
    text = trafilatura.extract(r.text, include_comments=False, include_tables=True) or ""
    return text.strip() or "the page had no readable text"


@tool()
def weather(place: str = "") -> str:
    """Current weather and today's forecast.

    Args:
        place: city name; leave empty for the user's current location
    """
    r = httpx.get(f"https://wttr.in/{urllib.parse.quote(place)}", params={"format": "j1"}, headers=UA, timeout=10)
    r.raise_for_status()
    d = r.json()
    cur = d["current_condition"][0]
    today = d["weather"][0]
    area = d.get("nearest_area", [{}])[0]
    name = ", ".join(x[0]["value"] for x in (area.get("areaName"), area.get("country")) if x) or place
    return (f"{name}: {cur['weatherDesc'][0]['value']}, {cur['temp_C']}°C (feels {cur['FeelsLikeC']}°C), "
            f"humidity {cur['humidity']}%, wind {cur['windspeedKmph']} km/h. "
            f"Today min {today['mintempC']}°C max {today['maxtempC']}°C, "
            f"chance of rain {max(int(h['chanceofrain']) for h in today['hourly'])}%.")

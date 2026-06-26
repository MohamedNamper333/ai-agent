import json
import re
from urllib.parse import urlparse, quote
from core.utils import strip_html


class WebTools:
    @staticmethod
    def fetch_url(url: str, timeout: int = 30) -> str:
        """Fetch a URL and return the extracted text content."""
        try:
            import requests
            parsed = urlparse(url)
            if not parsed.scheme:
                url = "https://" + url
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            content_type = r.headers.get("content-type", "").lower()
            if "json" in content_type:
                try:
                    return json.dumps(r.json(), indent=2, ensure_ascii=False)[:8000]
                except Exception:
                    pass
            if "xml" in content_type or "rss" in content_type:
                return f"URL: {url}\nContent-Type: {content_type}\n\n{r.text[:5000]}"
            text = strip_html(r.text)
            if len(text) > 10000:
                text = text[:10000] + "\n... (truncated)"
            return f"URL: {url}\nStatus: {r.status_code}\nLength: {len(r.text)} bytes\n\n{text}"
        except ImportError:
            return "Error: requests library not installed (pip install requests)"
        except Exception as e:
            return f"Error fetching URL: {e}"

    @staticmethod
    def search_web(query: str, num_results: int = 5) -> str:
        """Search DuckDuckGo and return the top results as formatted text."""
        try:
            import requests

            encoded_query = quote(query)
            urls_to_try = [
                f"https://html.duckduckgo.com/html/?q={encoded_query}",
                f"https://lite.duckduckgo.com/lite/?q={encoded_query}",
            ]

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }

            results = []
            for url in urls_to_try:
                try:
                    r = requests.get(url, headers=headers, timeout=15)
                    r.raise_for_status()

                    result_pattern = r'<a[^>]*class="result__a"[^>]*>(.*?)</a>'
                    snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>'

                    titles = re.findall(result_pattern, r.text, re.DOTALL)
                    snippets = re.findall(snippet_pattern, r.text, re.DOTALL)

                    if not titles:
                        result_pattern2 = r'<a[^>]*href="[^"]*"[^>]*class="result__a"[^>]*>(.*?)</a>'
                        titles = re.findall(result_pattern2, r.text, re.DOTALL)

                    for i, title in enumerate(titles[:num_results]):
                        title_clean = re.sub(r'<[^>]+>', '', title).strip()
                        snippet_clean = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
                        if title_clean:
                            results.append({
                                "title": title_clean,
                                "snippet": snippet_clean,
                            })

                    if results:
                        break
                except Exception:
                    continue

            if not results:
                return f"No results found for: {query}"

            lines = [f"Search results for: {query}\n"]
            for i, result in enumerate(results[:num_results], 1):
                lines.append(f"{i}. {result['title']}")
                if result['snippet']:
                    lines.append(f"   {result['snippet']}")
                lines.append("")

            return "\n".join(lines).strip()

        except ImportError:
            return "Error: requests library not installed"
        except Exception as e:
            return f"Error searching web: {e}"

    @staticmethod
    def web_scrape(url: str, selector: str = "") -> str:
        """Scrape a URL with an optional CSS selector and return the text."""
        try:
            import requests
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if not parsed.scheme:
                url = "https://" + url

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }

            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()

            content_type = r.headers.get("content-type", "").lower()

            if selector:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(r.text, 'html.parser')
                    elements = soup.select(selector)
                    if elements:
                        text_parts = [el.get_text(strip=True) for el in elements]
                        result = f"Selector: {selector}\nElements found: {len(elements)}\n\n"
                        result += "\n\n".join(text_parts[:20])
                        return result[:8000]
                    else:
                        return f"No elements found for selector: {selector}"
                except ImportError:
                    pass

            if "json" in content_type:
                try:
                    data = r.json()
                    return json.dumps(data, indent=2, ensure_ascii=False)[:8000]
                except Exception:
                    pass

            text = strip_html(r.text)
            return f"URL: {url}\nStatus: {r.status_code}\n\n{text[:8000]}"

        except ImportError:
            return "Error: requests library not installed"
        except Exception as e:
            return f"Error scraping URL: {e}"

    @staticmethod
    def fetch_json(url: str) -> str:
        """Fetch a URL and return the parsed JSON as a formatted string."""
        try:
            import requests
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if not parsed.scheme:
                url = "https://" + url

            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            }

            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()

            data = r.json()
            result = json.dumps(data, indent=2, ensure_ascii=False)
            if len(result) > 10000:
                result = result[:10000] + "\n... (truncated)"

            return result

        except ImportError:
            return "Error: requests library not installed"
        except Exception as e:
            return f"Error fetching JSON: {e}"

    @staticmethod
    def rss_feed(url: str) -> str:
        """Fetch and parse an RSS feed and return the item list."""
        try:
            import requests
            from xml.etree import ElementTree

            parsed = urlparse(url)
            if not parsed.scheme:
                url = "https://" + url

            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()

            root = ElementTree.fromstring(r.text)

            items = []
            for item in root.iter('item'):
                title = item.find('title')
                link = item.find('link')
                description = item.find('description')

                items.append({
                    "title": title.text if title is not None else "",
                    "link": link.text if link is not None else "",
                    "description": (description.text[:200] + "...") if description is not None and description.text else "",
                })

            if not items:
                for item in root.iter('{http://www.w3.org/2005/Atom}entry'):
                    title = item.find('{http://www.w3.org/2005/Atom}title')
                    link = item.find('{http://www.w3.org/2005/Atom}link')
                    summary = item.find('{http://www.w3.org/2005/Atom}summary')

                    items.append({
                        "title": title.text if title is not None else "",
                        "link": link.get('href', '') if link is not None else "",
                        "description": (summary.text[:200] + "...") if summary is not None and summary.text else "",
                    })

            if not items:
                return f"No items found in feed: {url}"

            lines = [f"RSS Feed: {url}", f"Items: {len(items)}", ""]
            for i, item in enumerate(items[:10], 1):
                lines.append(f"{i}. {item['title']}")
                if item['link']:
                    lines.append(f"   Link: {item['link']}")
                if item['description']:
                    lines.append(f"   {item['description']}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return f"Error reading RSS feed: {e}"

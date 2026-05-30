import json
import re
from urllib.parse import urlparse


class WebTools:
    @staticmethod
    def fetch_url(url: str, timeout: int = 30) -> str:
        try:
            import requests
            parsed = urlparse(url)
            if not parsed.scheme:
                url = "https://" + url

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()

            content_type = r.headers.get("content-type", "").lower()
            if "json" in content_type:
                try:
                    data = r.json()
                    return json.dumps(data, indent=2, ensure_ascii=False)[:5000]
                except Exception:
                    pass

            text = r.text
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            max_chars = 8000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"

            return f"URL: {url}\nLength: {len(r.text)} bytes\n\n{text[:max_chars]}"

        except ImportError:
            return "Error: requests library not installed (pip install requests)"
        except Exception as e:
            return f"Error fetching URL: {e}"

    @staticmethod
    def search_web(query: str, num_results: int = 5) -> str:
        try:
            import requests
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()

            from html.parser import HTMLParser

            class ResultParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results = []
                    self._current = {}
                    self._in_result = False
                    self._in_link = False
                    self._in_snippet = False
                    self._skip_a = 0

                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                        self._in_link = True
                        self._current["url"] = attrs_dict.get("href", "")
                        self._current["title"] = ""
                    if tag == "a" and self._in_link:
                        pass
                    if tag == "a" and "badge" in attrs_dict.get("class", ""):
                        pass

                def handle_endtag(self, tag):
                    if tag == "a" and self._in_link:
                        self._in_link = False
                        if "title" in self._current:
                            self._in_result = True

                def handle_data(self, data):
                    if self._in_link:
                        self._current["title"] = self._current.get("title", "") + data

                def add_result(self, result_line):
                    parts = result_line.split(" | ", 2)
                    if len(parts) >= 2:
                        self.results.append({
                            "title": parts[0].strip(),
                            "snippet": parts[1].strip() if len(parts) > 2 else "",
                        })

            parser = ResultParser()
            parser.feed(r.text)

            if not parser.results:
                snippets = re.findall(
                    r'<a[^>]*class="result__a"[^>]*>(.*?)</a>\s*<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                    r.text, re.DOTALL,
                )
                for title, snippet in snippets[:num_results]:
                    title = re.sub(r'<[^>]+>', '', title).strip()
                    snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    parser.results.append({"title": title, "snippet": snippet})

            if not parser.results:
                return "No results found. Try a different search query."

            lines = [f"Search results for: {query}\n"]
            for i, result in enumerate(parser.results[:num_results], 1):
                t = result.get("title", "").strip()
                s = result.get("snippet", "").strip()
                if t:
                    lines.append(f"{i}. {t}")
                if s:
                    lines.append(f"   {s[:200]}")
                lines.append("")

            return "\n".join(lines).strip()

        except ImportError:
            return "Error: requests library not installed"
        except Exception as e:
            return f"Error searching web: {e}"

    @staticmethod
    def web_scrape(url: str, selector: str = "") -> str:
        try:
            from html.parser import HTMLParser
        except ImportError:
            return "Error: HTMLParser not available"

        content = WebTools.fetch_url(url, timeout=30)
        if content.startswith("Error"):
            return content
        return content[:5000]

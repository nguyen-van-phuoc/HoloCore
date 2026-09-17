import os
import re
import sys
import json
import urllib3
import requests
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

# Disable SSL warnings when scraping misconfigured HTTPS websites.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SearchTool:
    def __init__(
            self,
            serper_api_key: str = os.getenv("SERPER_API_KEY"),
            ollama_model: str = "qwen3:4b-instruct",
            ollama_host: str = "http://localhost:11434",
            max_workers: int = 3
    ):
        self.serper_api_key = serper_api_key
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host.rstrip('/')
        self.max_workers = max_workers

        # Configure the session with an automatic retry mechanism for network access.
        self.session = requests.Session()
        retries = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
        try:
            res = self.session.post(url, json=payload, timeout=60)
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except Exception as e:
            print(f"[Ollama Error] Cannot call the model '{self.ollama_model}': {e}", file=sys.stderr)
            raise e

    def _select_best_urls(self, query: str, search_results: list) -> list:
        formatted_list = ""
        for idx, item in enumerate(search_results, 1):
            formatted_list += f"ID [{idx}]\nTiêu đề: {item.get('title')}\nTóm tắt: {item.get('snippet')}\nLink: {item.get('link')}\n\n"
        prompt = f"""Here is a list of Google search results for the query: "{query}"
                    {formatted_list}
                    Task: Select up to 2 results (IDs) that come from the most credible sources, are most directly relevant to the query, and are NOT social media sites (Facebook, TikTok, Instagram, YouTube).
                    Output requirement: Return ONLY a JSON array in the format [1, 2], without any additional explanation."""
        try:
            response = self._call_ollama(prompt)
            match = re.search(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', response)
            if match:
                selected_ids = json.loads(match.group(0))
                selected_items = [search_results[i - 1] for i in selected_ids if 1 <= i <= len(search_results)]
                if selected_items:
                    return selected_items
        except Exception as e:
            print(f"[Pipeline] Lỗi LLM chọn URL (dùng fallback 2 URL đầu): {e}", file=sys.stderr)
        return search_results[:2]

    def _scrape_clean_text(self, url: str) -> str:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
        try:
            res = self.session.get(url, headers=headers, timeout=(4, 10), verify=False)
            res.raise_for_status()

            soup = BeautifulSoup(res.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'form']):
                element.decompose()

            text = soup.get_text(separator=' ', strip=True)
            return text[:6000] if len(text) > 6000 else text
        except Exception as e:
            print(f"[Scrape Warning] Cannot be scratched off {url}: {e}", file=sys.stderr)
            return ""

    def _process_single_url(self, query: str, idx: int, target: dict) -> dict:

        url = target.get('link')
        title = target.get('title')
        snippet = target.get('snippet')

        print(f"[Thread-{idx}] Scraping data: {url}", file=sys.stderr)
        raw_text = self._scrape_clean_text(url)

        if len(raw_text) > 200:
            print(f"[Thread-{idx}] Having Ollama summarize the content...", file=sys.stderr)
            prompt = f"""Here is the text scraped from the website:
                            Title: {title}
                            Source: {url}
                            Content:
                            ---
                            {raw_text}
                            ---
                            Briefly summarize the key information from the text above to directly answer the question: "{query}".
                            Requirements:
                            - Clearly extract specific metrics, timelines, and important data (if any).
                            - Remove promotional sentences and extraneous information.
                            - Present the information using bullet points."""
            try:
                summary = self._call_ollama(prompt)
            except Exception:
                summary = f"(Error calling Ollama for summarization; using original summary): {snippet}"
        else:
            summary = f"(Unable to load the full webpage; using a summary from Google): {snippet}"

        return {
            "idx": idx,
            "title": title,
            "url": url,
            "summary": summary
        }

    def search(self, query: str) -> str:
        """The main method that executes the entire pipeline."""
        query = query.encode().decode("unicode_escape")
        print(f"\n[Search Pipeline] Search: '{query}'...\n", file=sys.stderr)

        # 1. Call the Serper API to retrieve the list of results.
        search_url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "gl": "vi", "hl": "vi", "num": 5}, ensure_ascii=False)
        headers = {'X-API-KEY': self.serper_api_key, 'Content-Type': 'application/json'}

        try:
            response = self.session.post(search_url, headers=headers, data=payload, timeout=(5, 12))
            response.raise_for_status()
            results = response.json().get("organic", [])
            if not results:
                return "No relevant information found on the web."
        except Exception as e:
            return f"Google API lookup error: {e}"

        # 2. Ollama selects the best URLs.
        print("[Search Pipeline] Ollama is selecting optimal URLs...", file=sys.stderr)
        selected_targets = self._select_best_urls(query, results)

        # 3. PARALLEL EXECUTION: Simultaneously scrape and summarize selected URLs.
        print(f"[Search Pipeline] Launch {len(selected_targets)} concurrent processing thread...", file=sys.stderr)
        processed_results = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(selected_targets))) as executor:
            futures = [
                executor.submit(self._process_single_url, query, idx, target)
                for idx, target in enumerate(selected_targets, 1)
            ]

            for future in as_completed(futures):
                try:
                    data = future.result()
                    processed_results.append(data)
                except Exception as exc:
                    print(f"[Pipeline Error] Faulty stream: {exc}", file=sys.stderr)

        # Reorder the results according to the original priority order.
        processed_results.sort(key=lambda x: x['idx'])

        # 4. Compile comprehensive information for the primary LLM.
        final_context = "TInformation simultaneously scraped and summarized from the Internet.:\n\n"
        for item in processed_results:
            final_context += f"=== SOURCE [{item['idx']}]: {item['title']} ===\n"
            final_context += f"Link: {item['url']}\n"
            final_context += f"Summary:\n{item['summary']}\n\n"

        return final_context


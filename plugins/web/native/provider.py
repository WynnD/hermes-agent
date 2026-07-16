"""Native local URL content extraction provider.

This backend gives Hermes a no-API-key ``web_extract`` path. It fetches pages
with httpx, extracts article text locally with Trafilatura/readability when
those optional deps are installed, then falls back to BeautifulSoup text
cleanup. Search and crawl remain delegated to the existing providers.
"""

from __future__ import annotations

import logging
from html import unescape
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_USER_AGENT = "HermesAgent/0.14 (+https://github.com/NousResearch/hermes-agent)"
_MAX_HTML_BYTES = 2_000_000
_MAX_PDF_BYTES = 50_000_000
_DEFAULT_TIMEOUT = 30.0


def _ensure_native_deps() -> None:
    """Install optional native extraction deps when lazy installs are allowed."""
    try:
        from tools.lazy_deps import FeatureUnavailable, ensure

        ensure("search.native", prompt=False)
    except ImportError:
        # Source/dev checkouts may not have lazy_deps importable during isolated
        # tests. The fallback extractor below still works without optional deps.
        return
    except FeatureUnavailable as exc:
        logger.debug("Native extractor optional deps unavailable: %s", exc)
    except Exception as exc:  # noqa: BLE001 - extraction should degrade, not die
        logger.debug("Native extractor lazy dependency check failed: %s", exc)


def _fetch_html(url: str) -> Tuple[str, str]:
    """Fetch a URL and return ``(text, content_type)``.

    Raises httpx exceptions to the caller, which converts them into per-URL
    error objects. ``web_extract_tool`` already performed SSRF checks before
    dispatching here.
    """
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
    }
    with httpx.Client(follow_redirects=True, timeout=_DEFAULT_TIMEOUT, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        max_bytes = _MAX_PDF_BYTES if _looks_like_pdf(url, content_type) else _MAX_HTML_BYTES
        content = response.content[: max_bytes + 1]
        if len(content) > max_bytes:
            kind = "PDF" if _looks_like_pdf(url, content_type) else "HTML"
            raise ValueError(f"response exceeds {max_bytes} byte native {kind} extraction limit")
        if _looks_like_pdf(url, content_type, content):
            return _extract_pdf_text(content, url), content_type or "application/pdf"
        response._content = content  # keep httpx charset detection on the capped bytes
        return response.text, content_type


def _looks_like_pdf(url: str, content_type: str = "", content: bytes = b"") -> bool:
    content_type_l = content_type.lower()
    if "application/pdf" in content_type_l or "+pdf" in content_type_l:
        return True
    if urlparse(url).path.lower().endswith(".pdf"):
        return True
    return content.startswith(b"%PDF-")


def _extract_pdf_text(content: bytes, url: str) -> str:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires PyMuPDF; install the native-web extra or enable lazy deps") from exc

    try:
        with pymupdf.open(stream=content, filetype="pdf") as doc:
            title = (doc.metadata or {}).get("title") or ""
            lines = []
            if title.strip():
                lines.append(f"# {title.strip()}")
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                lines.append(f"\n## Page {index}\n\n{text}")
        return "\n".join(lines).strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"PDF text extraction failed for {url}: {exc}") from exc


def _extract_with_trafilatura(html: str, url: str) -> Optional[Tuple[str, str, str]]:
    try:
        import trafilatura
    except ImportError:
        return None

    try:
        metadata = trafilatura.extract_metadata(html, default_url=url)
        content = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            include_images=False,
            favor_precision=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Trafilatura extraction failed for %s: %s", url, exc)
        return None

    if not content or not content.strip():
        return None
    title = ""
    if metadata is not None:
        title = getattr(metadata, "title", "") or ""
    return title.strip(), content.strip(), "trafilatura"


def _extract_with_readability(html: str, url: str) -> Optional[Tuple[str, str, str]]:
    try:
        from readability import Document
        from markdownify import markdownify as md
    except ImportError:
        return None

    try:
        doc = Document(html, url=url)
        title = unescape(doc.short_title() or doc.title() or "").strip()
        summary_html = doc.summary(html_partial=True)
        content = md(summary_html, heading_style="ATX").strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("readability extraction failed for %s: %s", url, exc)
        return None

    if not content:
        return None
    return title, content, "readability"


def _extract_with_bs4(html: str, url: str) -> Tuple[str, str, str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Last-ditch stdlib-ish cleanup if lazy deps were disabled/unavailable.
        text = _strip_html_tags(html)
        return "", text, "html-regex"

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    heading = soup.find(["h1", "h2"])
    if heading and heading.get_text(strip=True):
        title = heading.get_text(" ", strip=True)

    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content = "\n\n".join(lines)
    return title, content, "beautifulsoup"


def _strip_html_tags(html: str) -> str:
    import re

    without_scripts = re.sub(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\\1>", " ", html)
    without_tags = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
    return " ".join(unescape(without_tags).split())


def _extract_document(html: str, url: str, content_type: str = "") -> Tuple[str, str, str]:
    content_type_l = content_type.lower()
    # Plain text endpoints and PDF text extracted upstream do not need article extraction.
    if content_type_l.startswith("text/plain"):
        return "", html.strip(), "plain-text"
    if "application/pdf" in content_type_l or "+pdf" in content_type_l:
        title = next((line[2:].strip() for line in html.splitlines() if line.startswith("# ")), "")
        return title, html.strip(), "pymupdf"

    for extractor in (_extract_with_trafilatura, _extract_with_readability):
        result = extractor(html, url)
        if result and result[1].strip():
            return result
    return _extract_with_bs4(html, url)


class NativeWebSearchProvider(WebSearchProvider):
    """Local no-key URL extraction provider."""

    @property
    def name(self) -> str:
        return "native"

    @property
    def display_name(self) -> str:
        return "Native Extractor"

    def is_available(self) -> bool:
        # httpx is a core Hermes dependency and BeautifulSoup/trafilatura can be
        # lazy-installed on first use. No API key required.
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        _ensure_native_deps()
        results: List[Dict[str, Any]] = []
        for url in urls:
            try:
                from tools.interrupt import is_interrupted

                if is_interrupted():
                    results.append({"url": url, "title": "", "content": "", "raw_content": "", "error": "Interrupted"})
                    continue
            except Exception:
                pass

            try:
                html, content_type = _fetch_html(url)
                title, content, extractor = _extract_document(html, url, content_type)
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": {
                            "sourceURL": url,
                            "title": title,
                            "extractor": extractor,
                            "content_type": content_type,
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Native extract failed for %s: %s", url, exc)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": f"Native extract failed: {exc}",
                        "metadata": {"sourceURL": url, "extractor": "native"},
                    }
                )
        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": "No API key; local article extraction for web_extract only.",
            "env_vars": [],
        }

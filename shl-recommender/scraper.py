"""
SHL product catalog scraper: Individual Test Solutions only (type=1).
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

CATALOG_ENTRY_URL = "https://www.shl.com/solutions/products/product-catalog/"
CATALOG_RESOLVES_TO = "https://www.shl.com/products/product-catalog/"
BASE_HOST = "https://www.shl.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

INDIVIDUAL_TYPE = 1
PAGE_SIZE = 12
MAX_RETRIES = 4
BACKOFF_SEC = 1.5
DETAIL_WORKERS = 6

VALID_TEST_KEYS = frozenset("ABCDEKS")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _fetch_url(session: requests.Session, url: str, timeout: int = 45) -> str | None:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except (requests.RequestException, OSError) as e:
            last_err = e
            time.sleep(BACKOFF_SEC * (attempt + 1))
    if last_err:
        raise last_err
    return None


def _absolute_url(href: str) -> str:
    return urljoin(BASE_HOST + "/", href.lstrip("/"))


def _parse_list_row(tr: Any) -> dict[str, Any] | None:
    tds = tr.find_all("td")
    if len(tds) < 4:
        return None
    link = tds[0].find("a", href=True)
    if not link:
        return None
    name = " ".join(link.get_text(strip=True).split())
    href = link["href"]
    url = _absolute_url(href)
    remote = bool(tds[1].select_one("span.catalogue__circle.-yes"))
    adaptive = bool(tds[2].select_one("span.catalogue__circle.-yes"))
    keys: list[str] = []
    for span in tds[3].select("span.product-catalogue__key"):
        t = span.get_text(strip=True).upper()
        if t in VALID_TEST_KEYS:
            keys.append(t)
    test_type = ",".join(sorted(set(keys))) if keys else ""
    return {
        "name": name,
        "url": url,
        "description": "",
        "test_type": test_type,
        "remote_testing": remote,
        "adaptive": adaptive,
        "duration": "",
    }


def _parse_list_page(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        th = table.find("th", class_=re.compile(r"custom__table-heading__title"))
        if not th:
            continue
        title = th.get_text(strip=True)
        if "Individual Test Solutions" not in title:
            continue
        for tr in table.find_all("tr", attrs={"data-entity-id": True}):
            row = _parse_list_row(tr)
            if row:
                out.append(row)
        break
    return out


def _max_start_for_type(html: str, type_id: int) -> int:
    soup = BeautifulSoup(html, "html.parser")
    max_start = 0
    for a in soup.select("a.pagination__link[href]"):
        href = a.get("href", "")
        if f"type={type_id}" not in href:
            continue
        m = re.search(r"[?&]start=(\d+)", href)
        if m:
            max_start = max(max_start, int(m.group(1)))
    return max_start


def _parse_detail(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    description = ""
    duration = ""

    desc_h4 = None
    for h4 in soup.find_all("h4"):
        if h4.get_text(strip=True).lower() == "description":
            desc_h4 = h4
            break
    if desc_h4:
        parent = desc_h4.find_parent("div", class_=re.compile(r"product-catalogue-training-calendar__row"))
        if parent:
            p = parent.find("p")
            if p:
                description = " ".join(p.get_text(strip=True).split())

    if not description:
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            description = " ".join(meta["content"].strip().split())

    for h4 in soup.find_all("h4"):
        if "assessment length" in h4.get_text(strip=True).lower():
            row = h4.find_parent("div", class_=re.compile(r"product-catalogue-training-calendar__row"))
            if row:
                p = row.find("p")
                if p:
                    txt = p.get_text(" ", strip=True)
                    m = re.search(
                        r"Approximate\s+Completion\s+Time\s+in\s+minutes\s*=\s*(\d+)",
                        txt,
                        re.I,
                    )
                    if m:
                        duration = f"{m.group(1)} minutes"
                    elif "minute" in txt.lower():
                        duration = txt
            break

    return description, duration


def _fetch_detail(session: requests.Session, item: dict[str, Any]) -> dict[str, Any]:
    try:
        html = _fetch_url(session, item["url"], timeout=60)
        if not html:
            return item
        desc, dur = _parse_detail(html)
        if desc:
            item["description"] = desc
        if dur:
            item["duration"] = dur
    except Exception:
        pass
    return item


def scrape_individual_tests(
    *,
    fetch_details: bool = True,
    detail_workers: int = DETAIL_WORKERS,
) -> list[dict[str, Any]]:
    session = _session()
    list_url = CATALOG_ENTRY_URL
    first_html = _fetch_url(session, list_url)
    if not first_html:
        return []

    max_start = _max_start_for_type(first_html, INDIVIDUAL_TYPE)
    all_rows: list[dict[str, Any]] = []

    seen_urls: set[str] = set()
    starts = [0] + list(range(PAGE_SIZE, max_start + PAGE_SIZE, PAGE_SIZE))

    for start in starts:
        page_url = (
            f"{CATALOG_RESOLVES_TO}?start={start}&type={INDIVIDUAL_TYPE}"
            if start > 0
            else f"{CATALOG_RESOLVES_TO}?type={INDIVIDUAL_TYPE}"
        )
        html = first_html if start == 0 else _fetch_url(session, page_url)
        if not html:
            continue
        rows = _parse_list_page(html)
        if not rows:
            continue
        for r in rows:
            u = r["url"]
            if u not in seen_urls:
                seen_urls.add(u)
                all_rows.append(r)
        if start == 0:
            first_html = ""

    if fetch_details and all_rows:
        workers = max(1, min(detail_workers, len(all_rows)))

        def worker(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            sess = _session()
            return [_fetch_detail(sess, dict(x)) for x in batch]

        chunk = max(1, len(all_rows) // workers + 1)
        batches = [all_rows[i : i + chunk] for i in range(0, len(all_rows), chunk)]
        merged: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(worker, b) for b in batches]
            for fut in as_completed(futs):
                merged.extend(fut.result())
        by_url = {x["url"]: x for x in merged}
        all_rows = [by_url.get(r["url"], r) for r in all_rows]

    for r in all_rows:
        r["url"] = _canonical_shl_url(r["url"])
    return all_rows


def _canonical_shl_url(url: str) -> str:
    p = urlparse(url)
    if not p.scheme:
        return _absolute_url(url)
    netloc = p.netloc.lower()
    path = p.path or "/"
    if netloc.endswith("shl.com"):
        return f"https://www.shl.com{path}"
    return url


def save_catalog(items: list[dict[str, Any]], path: str | Path | None = None) -> Path:
    base = Path(__file__).resolve().parent
    out = Path(path) if path else base / "catalog.json"
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_scraper(output_path: str | Path | None = None) -> Path:
    items = scrape_individual_tests()
    return save_catalog(items, output_path)


if __name__ == "__main__":
    p = run_scraper()
    print(f"Wrote {p} with {len(json.loads(p.read_text(encoding='utf-8')))} assessments")

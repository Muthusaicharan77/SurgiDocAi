"""Download the guideline corpus (configs/guidelines.json) from Europe PMC.

    python scripts/fetch_guidelines.py

For each article: full-text JATS XML and its metadata (title, authors, journal,
licence) into data/guidelines/. Refuses any article whose licence is not CC BY,
so only openly licensed text ever enters the index. Texts are not committed.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ALLOWED_LICENSES = {"cc by"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def metadata(pmcid: str) -> dict:
    query = urllib.parse.urlencode(
        {"query": f"PMCID:{pmcid}", "format": "json", "resultType": "core"})
    hits = json.loads(get(f"{API}/search?{query}"))["resultList"]["result"]
    if not hits:
        raise SystemExit(f"{pmcid}: not found")
    r = hits[0]
    journal = r.get("journalInfo", {})
    return {
        "pmcid": pmcid,
        "title": r["title"],
        "authors": r.get("authorString", ""),
        "journal": journal.get("journal", {}).get("isoabbreviation", ""),
        "year": r.get("pubYear", ""),
        "doi": r.get("doi", ""),
        "license": r.get("license", ""),
        "url": f"https://europepmc.org/article/PMC/{pmcid}",
    }


def main() -> None:
    out = Path("data/guidelines")
    out.mkdir(parents=True, exist_ok=True)
    for entry in json.loads(Path("configs/guidelines.json").read_text()):
        meta = {"key": entry["key"], **metadata(entry["pmcid"])}
        if meta["license"].lower() not in ALLOWED_LICENSES:
            raise SystemExit(f"{entry['key']}: licence '{meta['license']}' not allowed")
        xml = get(f"{API}/{entry['pmcid']}/fullTextXML")
        ET.fromstring(xml)  # fail early on a broken download
        (out / f"{entry['key']}.xml").write_bytes(xml)
        (out / f"{entry['key']}.json").write_text(json.dumps(meta, indent=2))
        print(f"{entry['key']:16s} {meta['license']:6s} {meta['title'][:70]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Regenerate data/papers_raw.json from the Semantic Scholar Graph API (real abstracts, real reference lists).

    python fetch_papers.py                    # uses data/seed_titles.tsv, writes data/papers_raw.json
    S2_API_KEY=xxxx python fetch_papers.py    # recommended: higher rate limit (free key from semanticscholar.org/product/api)
    python fetch_papers.py --seeds data/seed_titles.tsv --out data/papers_raw.json --cache data/s2_cache.json

Then rebuild the knowledge state:   python -m pefkg build

What it does
  1. For every seed title, call  /graph/v1/paper/search/match  to resolve the Semantic Scholar paperId.
  2. Fetch title, year, authors, abstract (falls back to the TLDR if no abstract) and externalIds.
  3. Fetch the full reference list of each paper and KEEP ONLY references that are themselves seeds
     -> in-corpus citation edges (the corpus is deliberately small; depth over breadth).
  4. Year = first arXiv appearance when an arXiv id exists (so 'BERT' is 2018, not its 2019 venue year),
     otherwise the Semantic Scholar year. This keeps citation edges consistent in time.
Only the standard library is used. Responses are cached so the script can be re-run/resumed.
"""
import argparse, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request

API = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,year,abstract,authors,venue,externalIds,tldr"


# ---------------------------------------------------------------- pure parsing helpers (unit-tested)
def arxiv_year(external_ids):
    """'1810.04805' -> 2018 ; old-style ids and missing ids -> None."""
    aid = (external_ids or {}).get("ArXiv") or ""
    m = re.match(r"^(\d{2})(\d{2})\.\d+", aid)
    return 2000 + int(m.group(1)) if m else None


def parse_match(resp):
    data = (resp or {}).get("data") or []
    return data[0] if data else None


def parse_paper(rec, short_id, short, seed_title):
    abstract = rec.get("abstract")
    source = "semantic_scholar"
    if not abstract:
        tl = (rec.get("tldr") or {}).get("text")
        abstract, source = (tl, "semantic_scholar_tldr") if tl else ("", "missing")
    authors = [a.get("name") for a in (rec.get("authors") or []) if a.get("name")]
    return {"id": short_id, "short": short, "s2_paperId": rec.get("paperId"), "title": rec.get("title") or seed_title,
            "year": arxiv_year(rec.get("externalIds")) or rec.get("year"), "authors": authors[:1],
            "all_authors": authors, "venue": rec.get("venue"), "abstract": abstract,
            "abstract_source": source, "reference_source": "semantic_scholar"}


def refs_from(resp):
    """Return ([paperIds], next_offset_or_None) from a /references response."""
    ids = [(d.get("citedPaper") or {}).get("paperId") for d in (resp or {}).get("data", [])]
    return [i for i in ids if i], (resp or {}).get("next")


# ---------------------------------------------------------------- HTTP
def http_get(path, params, key=None, retries=6):
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "pefkg-fetch/1.0"}
    if key:
        headers["x-api-key"] = key
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt + 1
                print(f"  HTTP {e.code}, retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if e.code == 404:
                return None
            raise
        except urllib.error.URLError as e:
            print(f"  network error {e}; retrying", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url}")


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--seeds", default=os.path.join(root, "data", "seed_titles.tsv"))
    ap.add_argument("--out", default=os.path.join(root, "data", "papers_raw.json"))
    ap.add_argument("--cache", default=os.path.join(root, "data", "s2_cache.json"))
    ap.add_argument("--delay", type=float, default=None, help="seconds between calls (default 1.1 with key, 3.5 without)")
    args = ap.parse_args()
    key = os.environ.get("S2_API_KEY")
    delay = args.delay if args.delay is not None else (1.1 if key else 3.5)

    seeds = []
    with open(args.seeds) as f:
        next(f)
        for line in f:
            pid, short, year, title = line.rstrip("\n").split("\t")
            seeds.append((pid, short, int(year), title))
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}

    def save():
        json.dump(cache, open(args.cache, "w"))

    # 1+2: resolve ids and fetch metadata
    for pid, short, year, title in seeds:
        if pid in cache and cache[pid].get("rec"):
            continue
        print(f"[match] {pid}: {title}")
        m = parse_match(http_get("/paper/search/match", {"query": title, "fields": FIELDS}, key))
        time.sleep(delay)
        if not m:
            print(f"  !! no match for {title!r}", file=sys.stderr)
            cache[pid] = {"rec": None}
            continue
        if abs((arxiv_year(m.get("externalIds")) or m.get("year") or year) - year) > 1:
            print(f"  !! year mismatch: expected ~{year}, got {m.get('year')} - check '{m.get('title')}'", file=sys.stderr)
        cache[pid] = {"rec": m}
        save()

    # 3: references restricted to the corpus
    s2_to_local = {c["rec"]["paperId"]: pid for pid, c in cache.items() if c.get("rec")}
    for pid, c in cache.items():
        if not c.get("rec") or "refs" in c:
            continue
        print(f"[refs ] {pid}")
        refs, offset = [], 0
        while offset is not None:
            resp = http_get(f"/paper/{c['rec']['paperId']}/references", {"fields": "paperId", "limit": 1000, "offset": offset}, key)
            time.sleep(delay)
            ids, offset = refs_from(resp)
            refs += ids
        c["refs"] = [s2_to_local[i] for i in refs if i in s2_to_local and s2_to_local[i] != pid]
        save()

    papers = []
    for pid, short, year, title in seeds:
        c = cache.get(pid)
        if not c or not c.get("rec"):
            print(f"skipping {pid} (unresolved)", file=sys.stderr)
            continue
        p = parse_paper(c["rec"], pid, short, title)
        p["references"] = c.get("refs", [])
        papers.append(p)
    json.dump({"topic": "Parameter-efficient fine-tuning (PEFT) of language models",
               "source": "Semantic Scholar Graph API via fetch_papers.py", "papers": papers}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: {len(papers)} papers, {sum(len(p['references']) for p in papers)} in-corpus citation edges")


if __name__ == "__main__":
    main()

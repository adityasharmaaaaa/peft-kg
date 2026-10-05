"""
Build the knowledge state from raw paper records.

Pipeline
  1. extract.analyze_paper()      text -> mechanisms / problems / concepts / models / benchmarks (+ roles)
  2. type_citation()              raw citation (A cites B) -> typed relation using the signatures of A and B
  3. derived structure            influence, tiers, mechanism lineages, mechanism x problem coverage, IDF
  4. serialise                    knowledge_state/kg.json  (+ kg_summary.md for humans)

The raw citation list says only *that* A cites B. The schema decides *why it matters*:
EXTENDS / BUILDS_ON / COMPARES_WITH_PAPER / MOTIVATED_BY / USES_BACKBONE / ... (see ontology.RELATION_TYPES).
"""
import json, math, os, datetime
from collections import defaultdict, deque
from . import ontology as O
from .extract import analyze_paper

PERIODS = [("<=2020", lambda y: y <= 2020), ("2021-2022", lambda y: 2021 <= y <= 2022), (">=2023", lambda y: y >= 2023)]


def _idf(df, n):
    return round(math.log((n + 1) / (df + 1)) + 1, 4)


def _sig(paper, role):
    return {m for m, v in paper["signature"]["mechanisms"].items() if v["role"] == role}


def type_citation(A, B, specific_concepts, specific_problems):
    """Return (relation, evidence) for 'A cites B' (A newer)."""
    if A["contribution"] == "SURVEY":
        return "SURVEYS", "A is a survey covering B"
    if B["contribution"] == "BENCHMARK":
        shared = set(A["signature"]["benchmarks"]) & {b for b, v in B["signature"]["benchmarks"].items() if v["in_title"]}
        if shared:
            return "USES_BENCHMARK", "A reports results on " + ", ".join(sorted(shared))
        return "BACKGROUND", "cites a benchmark paper without evaluating on it"
    if B["contribution"] == "FOUNDATION_MODEL":
        intro = {m for m, v in B["signature"]["models"].items() if v["in_title"] or v["in_propose_sentence"]}
        shared = set(A["signature"]["models"]) & intro
        if shared:
            return "USES_BACKBONE", "A adapts/evaluates " + ", ".join(sorted(shared)) + " introduced by B"
        return "BACKGROUND", "foundation-model background"
    if B["contribution"] == "SURVEY":
        return "BACKGROUND", "cites a survey"

    Ap, Au, Ac = _sig(A, "PROPOSES"), _sig(A, "USES"), _sig(A, "COMPARES_WITH")
    Bp = _sig(B, "PROPOSES")
    if Ap & Bp:
        return "EXTENDS", "both propose " + ", ".join(sorted(Ap & Bp))
    if Bp & Au:
        return "BUILDS_ON", "A uses " + ", ".join(sorted(Bp & Au)) + " that B proposes"
    if Bp & Ac:
        return "COMPARES_WITH_PAPER", "A contrasts itself with " + ", ".join(sorted(Bp & Ac)) + " from B"
    if B["contribution"] == "ANALYSIS" or any(c["role"] == "ESTABLISHES" for c in B["signature"]["concepts"].values()):
        est = {c for c, v in B["signature"]["concepts"].items() if v["role"] == "ESTABLISHES" and c in specific_concepts}
        con = est & set(A["signature"]["concepts"])
        if con:
            return "MOTIVATED_BY", "A relies on concept " + ", ".join(sorted(con)) + " established by B"
        inv = _sig(B, "USES") | _sig(B, "COMPARES_WITH")
        shared = inv & (Ap | Au | Ac)
        if B["contribution"] == "ANALYSIS" and shared:
            return "MOTIVATED_BY", "A uses " + ", ".join(sorted(shared)) + " whose behaviour B analyses"
    # alternative approach attacking the same problem
    sp = set(A["signature"]["problems"]) & set(B["signature"]["problems"])
    spec = sp & specific_problems
    if Bp and (spec or len(sp) >= 2) and not (Ap & Bp):
        return "COMPARES_WITH_PAPER", "different mechanism, shared problem(s): " + ", ".join(sorted(spec or sp))
    return "BACKGROUND", "citation with no structural link in the schema"


def build(raw_path, out_dir, verbose=True):
    raw = json.load(open(raw_path))
    plist = raw["papers"]
    N = len(plist)
    ids = {p["id"] for p in plist}

    # ---------------- 1. extraction ---------------------------------------------------
    papers = {}
    for p in plist:
        ex = analyze_paper(p["title"], p["abstract"])
        papers[p["id"]] = {
            "id": p["id"], "short": p.get("short") or p["title"].split(":")[0], "title": p["title"], "year": p["year"],
            "authors": p.get("authors", []), "contribution": ex["contribution"],
            "claim_sentence": ex["claim_sentence"], "abstract": p["abstract"],
            "abstract_source": p.get("abstract_source", "unknown"), "reference_source": p.get("reference_source", "unknown"),
            "signature": {k: ex[k] for k in ("mechanisms", "problems", "concepts", "models", "benchmarks")},
        }

    # document frequencies (non-survey papers) -> IDF and 'specific' flags
    nonsurvey = [p for p in papers.values() if p["contribution"] != "SURVEY"]
    M = len(nonsurvey)
    df = defaultdict(int)
    for p in nonsurvey:
        s = p["signature"]
        for m in s["mechanisms"]: df[f"mechanism:{m}"] += 1
        for k in ("problems", "concepts", "models", "benchmarks"):
            for e in s[k]: df[f"{k[:-1]}:{e}"] += 1
    idf = {k: _idf(v, M) for k, v in df.items()}
    specific_concepts = {c for c in O.CONCEPTS if not O.CONCEPTS[c].get("generic") and df[f"concept:{c}"] / M < 0.25}
    specific_problems = {q for q in O.PROBLEMS if df[f"problem:{q}"] / M < 0.20}

    # ---------------- 2. typed citation edges ----------------------------------------
    edges, dropped = [], []
    for p in plist:
        A = papers[p["id"]]
        for rid in p.get("references", []):
            if rid not in ids:
                continue
            B = papers[rid]
            if B["year"] > A["year"]:
                dropped.append((A["id"], B["id"], "cites a later paper")); continue
            rel, ev = type_citation(A, B, specific_concepts, specific_problems)
            edges.append({"source": f"paper:{A['id']}", "relation": rel, "target": f"paper:{B['id']}", "evidence": ev})

    # ---------------- entity edges ----------------------------------------------------
    for pid, p in papers.items():
        src = f"paper:{pid}"
        for m, v in p["signature"]["mechanisms"].items():
            edges.append({"source": src, "relation": v["role"], "target": f"mechanism:{m}",
                          "evidence": f'{v["evidence_type"]}: "{v["match"]}"'})
        for q, v in p["signature"]["problems"].items():
            edges.append({"source": src, "relation": "ADDRESSES", "target": f"problem:{q}", "evidence": f'"{v["match"]}"'})
        for c, v in p["signature"]["concepts"].items():
            edges.append({"source": src, "relation": v["role"], "target": f"concept:{c}", "evidence": f'"{v["match"]}"'})
        for m, v in p["signature"]["models"].items():
            rel = "INTRODUCES_MODEL" if p["contribution"] == "FOUNDATION_MODEL" and (v["in_title"] or v["in_propose_sentence"]) else "USES_MODEL"
            edges.append({"source": src, "relation": rel, "target": f"model:{m}", "evidence": f'"{v["match"]}"'})
        for b, v in p["signature"]["benchmarks"].items():
            rel = "INTRODUCES_BENCHMARK" if p["contribution"] == "BENCHMARK" and v["in_title"] else "EVALUATED_ON"
            edges.append({"source": src, "relation": rel, "target": f"benchmark:{b}", "evidence": f'"{v["match"]}"'})
        for a in p["authors"]:
            edges.append({"source": src, "relation": "AUTHORED_BY", "target": f"author:{a}", "evidence": "lead author"})
    for m, spec in O.MECHANISMS.items():
        edges.append({"source": f"mechanism:{m}", "relation": "SUBTYPE_OF", "target": f"family:{spec['family']}", "evidence": "taxonomy"})
    for c, ms in O.CONCEPT_UNDERPINS.items():
        for m in ms:
            edges.append({"source": f"concept:{c}", "relation": "UNDERPINS", "target": f"mechanism:{m}", "evidence": "hand-authored background knowledge"})

    # ---------------- 3. derived structure -------------------------------------------
    prereq = defaultdict(set)         # A -> {B}: A requires B (strong relations + backbone)
    for e in edges:
        if e["source"].startswith("paper:") and e["target"].startswith("paper:") and \
           e["relation"] in (O.PREREQ_STRONG | {"USES_BACKBONE"}):
            prereq[e["source"][6:]].add(e["target"][6:])
    desc = defaultdict(set)           # B -> all papers that (transitively) require B
    rev = defaultdict(set)
    for a, bs in prereq.items():
        for b in bs: rev[b].add(a)
    for pid in papers:
        seen, dq = set(), deque([pid])
        while dq:
            x = dq.popleft()
            for y in rev[x]:
                if y not in seen: seen.add(y); dq.append(y)
        desc[pid] = seen
    cited_by = defaultdict(int)
    for e in edges:
        if e["source"].startswith("paper:") and e["target"].startswith("paper:"):
            cited_by[e["target"][6:]] += 1

    mech_nodes = {}
    for m, spec in O.MECHANISMS.items():
        prop = sorted([p for p in papers.values() if p["signature"]["mechanisms"].get(m, {}).get("role") == "PROPOSES"], key=lambda p: (p["year"], p["id"]))
        use = [p["id"] for p in papers.values() if p["signature"]["mechanisms"].get(m, {}).get("role") == "USES"]
        cmp_ = [p["id"] for p in papers.values() if p["signature"]["mechanisms"].get(m, {}).get("role") == "COMPARES_WITH"]
        by_period = {name: sum(1 for p in prop if fn(p["year"])) for name, fn in PERIODS}
        mech_nodes[m] = {"id": m, "label": spec["label"], "family": spec["family"], "description": spec["desc"],
                         "proposed_by": [p["id"] for p in prop], "first_in_corpus": prop[0]["id"] if prop else None,
                         "used_by": use, "compared_by": cmp_, "proposals_by_period": by_period}

    first_of = {m: v["first_in_corpus"] for m, v in mech_nodes.items() if v["first_in_corpus"]}
    for pid, p in papers.items():
        c = p["contribution"]
        proposes = _sig(p, "PROPOSES")
        if c == "METHOD":
            tier = "CORE_METHOD" if any(first_of.get(m) == pid for m in proposes) else "EXTENSION"
        else:
            tier = {"FOUNDATION_MODEL": "FOUNDATION", "BENCHMARK": "FOUNDATION", "ANALYSIS": "ANALYSIS", "SURVEY": "SURVEY"}.get(c, "OTHER")
        p["tier"] = tier
        p["cited_by_in_corpus"] = cited_by[pid]
        p["influence"] = len(desc[pid])           # papers that transitively require this one
        p["originates"] = sorted(m for m in proposes if first_of.get(m) == pid)

    # mechanism x problem coverage (METHOD papers only)
    matrix = defaultdict(lambda: defaultdict(lambda: {"proposes": [], "uses": []}))
    for pid, p in papers.items():
        if p["contribution"] != "METHOD": continue
        for m, v in p["signature"]["mechanisms"].items():
            if v["role"] not in ("PROPOSES", "USES"): continue
            for q in p["signature"]["problems"]:
                matrix[m][q]["proposes" if v["role"] == "PROPOSES" else "uses"].append(pid)

    problem_nodes = {}
    for q, spec in O.PROBLEMS.items():
        addr = [pid for pid, p in papers.items() if q in p["signature"]["problems"]]
        mech_counts = defaultdict(int)
        for m in matrix:
            if q in matrix[m]: mech_counts[m] = len(matrix[m][q]["proposes"]) + len(matrix[m][q]["uses"])
        problem_nodes[q] = {"id": q, "label": spec["label"], "addressed_by": sorted(addr, key=lambda x: papers[x]["year"]),
                            "mechanisms_applied": dict(sorted(mech_counts.items(), key=lambda kv: -kv[1]))}

    concept_nodes = {c: {"id": c, "label": s["label"], "generic": bool(s.get("generic")),
                         "established_by": [pid for pid, p in papers.items() if p["signature"]["concepts"].get(c, {}).get("role") == "ESTABLISHES"],
                         "relied_on_by": [pid for pid, p in papers.items() if p["signature"]["concepts"].get(c, {}).get("role") == "RELIES_ON"],
                         "underpins": O.CONCEPT_UNDERPINS.get(c, [])} for c, s in O.CONCEPTS.items()}
    model_nodes = {m: {"id": m, "label": s["label"], "used_by": [pid for pid, p in papers.items() if m in p["signature"]["models"]]} for m, s in O.MODELS.items()}
    bench_nodes = {b: {"id": b, "label": s["label"], "used_by": [pid for pid, p in papers.items() if b in p["signature"]["benchmarks"]]} for b, s in O.BENCHMARKS.items()}
    author_nodes = defaultdict(lambda: {"papers": []})
    for pid, p in papers.items():
        for a in p["authors"]: author_nodes[a]["papers"].append(pid)

    kg = {
        "meta": {
            "schema_version": O.SCHEMA_VERSION, "domain": "Research paper onboarding",
            "topic": raw.get("topic"), "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "raw_source": raw.get("source"), "n_papers": N, "n_edges": len(edges),
            "abstract_sources": sorted({p["abstract_source"] for p in papers.values()}),
            "reference_sources": sorted({p["reference_source"] for p in papers.values()}),
            "dropped_citation_edges": dropped,
            "extraction": "rule-based lexicon + role rules (pefkg/ontology.py, pefkg/extract.py); no NER/RE library",
        },
        "schema": {"entity_types": O.ENTITY_TYPES, "relation_types": O.RELATION_TYPES, "mechanism_families": O.MECHANISM_FAMILIES,
                   "prerequisite_relations": sorted(O.PREREQ_STRONG), "foundation_relations": sorted(O.PREREQ_FOUNDATION)},
        "nodes": {
            "papers": papers, "mechanisms": mech_nodes, "problems": problem_nodes, "concepts": concept_nodes,
            "models": model_nodes, "benchmarks": bench_nodes, "authors": dict(author_nodes),
            "families": {f: {"id": f, "description": d, "mechanisms": [m for m, s in O.MECHANISMS.items() if s["family"] == f]} for f, d in O.MECHANISM_FAMILIES.items()},
        },
        "edges": edges,
        "derived": {
            "idf": idf, "n_nonsurvey_papers": M,
            "specific_concepts": sorted(specific_concepts), "specific_problems": sorted(specific_problems),
            "mechanism_problem_matrix": {m: dict(q) for m, q in matrix.items()},
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kg.json"), "w") as f:
        json.dump(kg, f, indent=1)
    from .summary import write_summary
    write_summary(kg, os.path.join(out_dir, "kg_summary.md"))
    if verbose:
        rels = defaultdict(int)
        for e in edges: rels[e["relation"]] += 1
        print(f"papers={N} edges={len(edges)} dropped={len(dropped)}")
        print("relations:", ", ".join(f"{k}={v}" for k, v in sorted(rels.items(), key=lambda kv: -kv[1])))
    return kg

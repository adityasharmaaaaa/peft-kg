"""Render knowledge_state/kg_summary.md - a human-readable view of kg.json."""
from collections import defaultdict


def write_summary(kg, path):
    N = kg["nodes"]
    papers = N["papers"]
    L = []
    w = L.append
    meta = kg["meta"]
    w("# Knowledge state summary\n")
    w(f"Topic: **{meta['topic']}**  \nPapers: **{meta['n_papers']}** | Edges: **{meta['n_edges']}** | Built: {meta['built_at']}  ")
    w(f"Abstract source: `{', '.join(meta['abstract_sources'])}` | Reference source: `{', '.join(meta['reference_sources'])}`\n")
    w("> This file is generated from `kg.json`. `kg.json` is the source of truth.\n")

    # relation counts
    rc = defaultdict(int)
    for e in kg["edges"]:
        rc[e["relation"]] += 1
    w("## Edge counts by relation\n")
    w("| relation | count | meaning |\n|---|---|---|")
    for r, c in sorted(rc.items(), key=lambda kv: -kv[1]):
        w(f"| {r} | {c} | {kg['schema']['relation_types'].get(r, '')[:110]} |")
    w("")

    # mechanisms by family
    w("## Mechanism taxonomy and who proposed what\n")
    for fam, fn in N["families"].items():
        w(f"### {fam}\n_{fn['description']}_\n")
        w("| mechanism | first in corpus | proposed by (chronological) | used by | proposals <=2020 / 21-22 / >=23 |\n|---|---|---|---|---|")
        for m in fn["mechanisms"]:
            mn = N["mechanisms"][m]
            pb = ", ".join(papers[p]["short"] for p in mn["proposed_by"]) or "-"
            ub = ", ".join(papers[p]["short"] for p in mn["used_by"][:6]) + ("..." if len(mn["used_by"]) > 6 else "")
            per = mn["proposals_by_period"]
            first = papers[mn["first_in_corpus"]]["short"] if mn["first_in_corpus"] else "-"
            w(f"| `{m}` {mn['label']} | {first} | {pb} | {ub or '-'} | {list(per.values())} |")
        w("")

    # papers by tier
    w("## Papers by tier (influence = number of papers that transitively require it)\n")
    for tier in ("FOUNDATION", "ANALYSIS", "CORE_METHOD", "EXTENSION", "SURVEY", "OTHER"):
        ps = sorted([p for p in papers.values() if p["tier"] == tier], key=lambda p: (p["year"], p["id"]))
        if not ps:
            continue
        w(f"### {tier} ({len(ps)})\n")
        w("| paper | year | proposes | addresses | influence |\n|---|---|---|---|---|")
        for p in ps:
            prop = ", ".join(m for m, v in p["signature"]["mechanisms"].items() if v["role"] == "PROPOSES") or "-"
            addr = ", ".join(list(p["signature"]["problems"])[:4]) or "-"
            w(f"| {p['short']} (`{p['id']}`) | {p['year']} | {prop} | {addr} | {p['influence']} |")
        w("")

    # lineage graph
    w("## Lineage graph (EXTENDS = solid, BUILDS_ON = dashed; method papers only)\n")
    w("```mermaid\ngraph LR")
    keep = {pid for pid, p in papers.items() if p["contribution"] == "METHOD"}
    for pid in sorted(keep):
        w(f'  {pid}["{papers[pid]["short"]} {papers[pid]["year"]}"]')
    for e in kg["edges"]:
        if e["relation"] in ("EXTENDS", "BUILDS_ON") and e["source"][6:] in keep and e["target"][6:] in keep:
            a, b = e["source"][6:], e["target"][6:]
            w(f"  {b} -->|ext| {a}" if e["relation"] == "EXTENDS" else f"  {b} -.->|uses| {a}")
    w("```\n")

    # coverage
    w("## Mechanism x problem coverage (method papers; count of papers proposing or using)\n")
    mat = kg["derived"]["mechanism_problem_matrix"]
    for m, row in sorted(mat.items()):
        cells = ", ".join(f"{q} ({len(v['proposes']) + len(v['uses'])})" for q, v in sorted(row.items(), key=lambda kv: -(len(kv[1]['proposes']) + len(kv[1]['uses']))))
        w(f"- **{m}**: {cells}")
    w("")
    w("## Problems and how many method papers attack them\n")
    for q, pn in sorted(N["problems"].items(), key=lambda kv: -len(kv[1]["addressed_by"])):
        w(f"- **{q}** ({pn['label']}): {len(pn['addressed_by'])} papers; top mechanisms: {', '.join(list(pn['mechanisms_applied'])[:4]) or '-'}")
    w("")
    open(path, "w").write("\n".join(L))

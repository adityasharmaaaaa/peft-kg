"""
Reasoning over the knowledge state for a NEW input.

The reasoner loads only `knowledge_state/kg.json` (the serialized state) plus the ontology
lexicon. It never touches the raw paper text again.

Steps
  1. understand()     map free text onto the ontology -> an "idea signature"
  2. closest_work()   rank method/analysis papers by IDF-weighted overlap with the signature
  3. novelty()        rule-based verdict from the mechanism x problem coverage structure
  4. reading_path()   prerequisite closure over typed citation edges, topologically ordered
  5. baselines()      what to compare against + the evaluation setup those papers share
  6. gaps()           empty cells / never-combined mechanisms adjacent to the idea
  7. momentum()       is each mechanism rising or settled in the corpus
Every output item carries the reason it was produced.
"""
import json, math
from collections import defaultdict, deque
from . import ontology as O
from .extract import analyze_input

TIER_ORDER = {"FOUNDATION": 0, "ANALYSIS": 0, "CORE_METHOD": 1, "EXTENSION": 2, "SURVEY": 9, "OTHER": 3}


class Reasoner:
    def __init__(self, kg_path):
        self.kg = json.load(open(kg_path))
        self.P = self.kg["nodes"]["papers"]
        self.MN = self.kg["nodes"]["mechanisms"]
        self.PN = self.kg["nodes"]["problems"]
        self.idf = self.kg["derived"]["idf"]
        self.matrix = self.kg["derived"]["mechanism_problem_matrix"]
        self.prereq_edges = [(e["source"][6:], e["relation"], e["target"][6:], e["evidence"]) for e in self.kg["edges"]
                             if e["source"].startswith("paper:") and e["target"].startswith("paper:")
                             and e["relation"] in (O.PREREQ_STRONG | O.PREREQ_FOUNDATION | {"USES_BACKBONE"})]
        self.method_ids = [p for p, v in self.P.items() if v["contribution"] == "METHOD"]

    # ------------------------------------------------------------------ helpers
    def _w(self, kind, eid):
        return self.idf.get(f"{kind}:{eid}", 1.0)

    def short(self, pid):
        return self.P[pid]["short"]

    def _mech(self, pid, roles=("PROPOSES", "USES")):
        return {m for m, v in self.P[pid]["signature"]["mechanisms"].items() if v["role"] in roles}

    def _prop(self, pid):
        return self._mech(pid, ("PROPOSES",))

    # ------------------------------------------------------------------ 1. understand
    def understand(self, text):
        a = analyze_input(text)
        idea = {m for m, v in a["mechanisms"].items() if v["role"] == "IN_IDEA"}
        foil = {m for m, v in a["mechanisms"].items() if v["role"] == "BASELINE_OR_FOIL"}
        n_rec = len(idea) + len(a["problems"])
        return {
            "raw": a, "mechanisms": idea, "foils": foil, "problems": set(a["problems"]),
            "concepts": set(a["concepts"]), "models": set(a["models"]), "benchmarks": set(a["benchmarks"]),
            "confidence": "high" if n_rec >= 4 else "medium" if n_rec >= 2 else "low",
        }

    # ------------------------------------------------------------------ 2. closest work
    def _paper_vec(self, pid):
        s = self.P[pid]["signature"]
        return ({m: (1.0 if v["role"] == "PROPOSES" else 0.5) for m, v in s["mechanisms"].items() if v["role"] in ("PROPOSES", "USES")},
                set(s["problems"]), set(s["concepts"]) | set(s["models"]))

    def score(self, u, pid):
        pm, pp, pcm = self._paper_vec(pid)
        dims = []
        if u["mechanisms"]:
            inter = sum(min(1.0, pm.get(m, 0)) * self._w("mechanism", m) for m in u["mechanisms"])
            union = sum(self._w("mechanism", m) for m in u["mechanisms"]) + sum(w * self._w("mechanism", m) for m, w in pm.items() if m not in u["mechanisms"])
            cont = inter / max(1e-9, sum(self._w("mechanism", m) for m in u["mechanisms"]))
            dims.append((0.55, 0.5 * inter / max(1e-9, union) + 0.5 * cont))
        if u["problems"]:
            inter = sum(self._w("problem", q) for q in u["problems"] & pp)
            union = sum(self._w("problem", q) for q in u["problems"] | pp)
            cont = inter / max(1e-9, sum(self._w("problem", q) for q in u["problems"]))
            dims.append((0.30, 0.5 * inter / max(1e-9, union) + 0.5 * cont))
        ic = u["concepts"] | u["models"]
        if ic:
            inter = len(ic & pcm)
            dims.append((0.15, inter / max(1, len(ic | pcm))))
        if not dims:
            return 0.0
        tw = sum(w for w, _ in dims)
        return sum(w * s for w, s in dims) / tw

    def closest_work(self, u, k=5):
        ranked = []
        for pid, p in self.P.items():
            if p["contribution"] not in ("METHOD", "ANALYSIS"):
                continue
            sc = self.score(u, pid)
            if sc > 0:
                ranked.append((sc, pid))
        ranked.sort(key=lambda t: (-t[0], self.P[t[1]]["year"]))
        out = []
        for sc, pid in ranked[:k]:
            pm, pp, _ = self._paper_vec(pid)
            shared_m = sorted(u["mechanisms"] & set(pm))
            shared_p = sorted(u["problems"] & pp)
            proposed_shared = sorted(u["mechanisms"] & self._prop(pid))
            missing = sorted(u["mechanisms"] - set(pm))
            extra = sorted(self._prop(pid) - u["mechanisms"])
            if u["mechanisms"] and not missing and shared_p:
                rel = "already contains every mechanism you list and attacks a problem you name"
            elif proposed_shared:
                rel = "proposes " + ", ".join(proposed_shared) + (f"; your idea additionally involves {', '.join(missing)}" if missing else "")
            elif shared_p:
                rel = "attacks the same problem(s) with a different mechanism"
            else:
                rel = "partial overlap"
            out.append({"paper": pid, "title": self.P[pid]["title"], "year": self.P[pid]["year"], "score": round(sc, 3),
                        "shared_mechanisms": shared_m, "shared_problems": shared_p,
                        "your_mechanisms_it_lacks": missing, "its_mechanisms_you_lack": extra, "relation_to_your_idea": rel})
        return out

    # ------------------------------------------------------------------ 3. novelty
    def novelty(self, u):
        Min, Pin = u["mechanisms"], u["problems"]
        rationale, untested = [], []
        if not Min:
            return {"verdict": "NOT_ASSESSED",
                    "explanation": "No specific mechanism was recognised in the input, so novelty of a technique cannot be judged. "
                                   "The reading path and problem landscape below are still produced.",
                    "rationale": [], "untested_cells": []}
        method_mech = {pid: self._mech(pid) for pid in self.method_ids}
        holders = {m: [pid for pid, ms in method_mech.items() if m in ms] for m in Min}
        full = [pid for pid, ms in method_mech.items() if Min <= ms]
        prob = lambda pid: set(self.P[pid]["signature"]["problems"])
        for m in sorted(Min):
            for q in sorted(Pin):
                cell = self.matrix.get(m, {}).get(q)
                n = len(cell["proposes"]) + len(cell["uses"]) if cell else 0
                if n == 0:
                    untested.append({"mechanism": m, "problem": q, "reason": "no method paper in the corpus applies this mechanism to this problem"})

        if full:
            best = min(full, key=lambda pid: (-len(Pin & prob(pid)), self.P[pid]["year"]))   # best problem match, earliest wins ties
            overlap = Pin & prob(best)
            if Pin and Pin <= prob(best):
                verdict = "ALREADY_COVERED"
                rationale.append(f"{self.short(best)} ({self.P[best]['year']}) already involves all of {sorted(Min)} and addresses all of {sorted(Pin)}.")
                orig = sorted(set(self.P[best]["originates"]) & Min)
                if orig:
                    rationale.append(f"{self.short(best)} is where {', '.join(orig)} first appears in the corpus: your description reads like that paper's own contribution. "
                                     "Later papers that also contain the mechanism are refinements of it, not independent prior art for your exact idea.")
                later = [p for p in full if p != best and self.P[p]["year"] >= self.P[best]["year"]]
                if later:
                    rationale.append("Other corpus papers containing the same mechanism set: " + ", ".join(self.short(p) for p in later[:4]) + ".")
            elif overlap:
                verdict = "INCREMENTAL"
                rationale.append(f"{self.short(best)} involves all of your mechanisms and addresses {sorted(overlap)}; "
                                 f"what remains open is {sorted(Pin - overlap)}.")
            elif Pin:
                verdict = "NOVEL_APPLICATION"
                rationale.append(f"Papers exist that use your full mechanism set ({', '.join(self.short(p) for p in full[:3])}), "
                                 f"but none addresses {sorted(Pin)}.")
            else:
                verdict = "MECHANISM_SET_EXISTS"
                rationale.append(f"Your mechanism set already appears in {', '.join(self.short(p) for p in full[:3])}; "
                                 "no problem was named, so the angle of attack cannot be judged.")
        else:
            missing_holders = [m for m in Min if not holders[m]]
            if missing_holders:
                verdict = "MECHANISM_NOT_IN_CORPUS"
                rationale.append(f"No method paper in the corpus involves {missing_holders}; the corpus cannot say whether it is new.")
            elif len(Min) >= 2:
                verdict = "NOVEL_COMBINATION"
                rationale.append(f"Each of {sorted(Min)} is covered by method papers, but no single paper combines all of them.")
                # maximal covered sub-combinations
                best_sub = []
                for pid, ms in method_mech.items():
                    sub = Min & ms
                    if len(sub) >= 2:
                        best_sub.append((len(sub), pid, sorted(sub)))
                best_sub.sort(key=lambda t: (-t[0], self.P[t[1]]["year"]))
                for n, pid, sub in best_sub[:3]:
                    rationale.append(f"{self.short(pid)} covers the sub-combination {sub} (missing {sorted(Min - set(sub))}).")
            else:
                verdict = "MECHANISM_NOT_IN_CORPUS"
                rationale.append("The mechanism has no holder in the method papers.")
        return {"verdict": verdict, "rationale": rationale, "untested_cells": untested[:8]}

    # ------------------------------------------------------------------ 4. reading path
    def reading_path(self, u, closest, level="beginner", max_papers=12):
        beginner = level != "expert"
        seeds = [c["paper"] for c in closest[:3]]
        # lineage anchors: where each of the user's mechanisms first appears in the corpus
        anchors = [self.MN[m]["first_in_corpus"] for m in u["mechanisms"] if self.MN[m]["first_in_corpus"]]
        roots = list(dict.fromkeys(seeds + anchors))
        adj = defaultdict(list)       # A -> [(B, rel, evidence)] ; A requires B
        for a, rel, b, ev in self.prereq_edges:
            if rel in O.PREREQ_FOUNDATION or rel == "USES_BACKBONE":
                if not beginner or self.P[b]["tier"] != "FOUNDATION":
                    continue
            adj[a].append((b, rel, ev))
        max_depth = 3 if beginner else 2          # experts only need the near lineage
        depth, via = {r: 0 for r in roots}, {}
        dq = deque(roots)
        while dq:
            a = dq.popleft()
            if depth[a] >= max_depth:
                continue
            for b, rel, ev in adj[a]:
                if self.P[b]["tier"] == "SURVEY":
                    continue
                if b not in depth:
                    depth[b] = depth[a] + 1
                    via[b] = (a, rel, ev)
                    dq.append(b)
        if not beginner:   # experts skip generic foundations unless they are the model family being adapted
            depth = {p: d for p, d in depth.items()
                     if p in roots
                     or (self.P[p]["tier"] == "FOUNDATION" and (set(self.P[p]["signature"]["models"]) & u["models"]))
                     or (self.P[p]["tier"] == "ANALYSIS" and d <= 1)
                     or self.P[p]["tier"] not in ("FOUNDATION", "ANALYSIS")}
        maxinf = max(1, max(self.P[p]["influence"] for p in depth))
        cscore = {c["paper"]: c["score"] for c in closest}

        def rel_score(p):
            return 0.4 / (1 + depth[p]) + 0.3 * cscore.get(p, 0) + 0.3 * self.P[p]["influence"] / maxinf
        keep = set(roots) | set(sorted((p for p in depth if p not in roots), key=lambda p: -rel_score(p))[:max(0, max_papers - len(roots))])
        # topological order among kept (requirement edges restricted to kept set)
        req = {p: {b for b, _, _ in adj[p] if b in keep} for p in keep}
        order, done = [], set()
        while len(order) < len(keep):
            ready = [p for p in keep if p not in done and req[p] <= done]
            if not ready:                                   # cycle guard (should not happen: edges go back in time)
                ready = [p for p in keep if p not in done]
            nxt = min(ready, key=lambda p: (p in seeds, TIER_ORDER.get(self.P[p]["tier"], 3), self.P[p]["year"], -self.P[p]["influence"], p))
            order.append(nxt); done.add(nxt)

        steps = []
        for i, pid in enumerate(order, 1):
            p = self.P[pid]
            if pid in seeds:
                sh = next(c for c in closest if c["paper"] == pid)
                why = f"Closest prior work to your idea: {sh['relation_to_your_idea']}."
                stage = "Closest prior work (read last, compare line by line)"
            elif pid in via:
                a, rel, ev = via[pid]
                if p["tier"] == "FOUNDATION" and rel in ("BACKGROUND", "USES_BACKBONE"):
                    why = (f"Foundation: {p['influence']} corpus papers build on it; needed to follow {self.short(a)}"
                           + (f" ({ev})." if rel == "USES_BACKBONE" else "."))
                else:
                    why = f"{self.short(a)} {rel.replace('_', ' ').lower()} {p['short']} ({ev})."
                stage = {"FOUNDATION": "Foundations", "ANALYSIS": "Empirical grounding"}.get(p["tier"], "Method lineage")
            else:
                why = "Where your mechanism first appears in the corpus."
                stage = "Method lineage"
            if pid in anchors and pid not in seeds:
                why = "Originates " + ", ".join(self.MN[m]["label"] for m in u["mechanisms"] if self.MN[m]["first_in_corpus"] == pid) + ", which your idea uses. " + why
            steps.append({"step": i, "stage": stage, "paper": pid, "title": p["title"], "year": p["year"], "tier": p["tier"],
                          "why": why, "read_after": sorted(req[pid])})
        return {"level": level, "steps": steps, "orientation": self._orientation(u)}

    def _orientation(self, u):
        best = None
        for pid, p in self.P.items():
            if p["contribution"] != "SURVEY":
                continue
            cov = {m for m, v in p["signature"]["mechanisms"].items() if v["role"] == "COVERS"}
            sc = (len(cov & u["mechanisms"]), p["year"])
            if best is None or sc > best[0]:
                best = (sc, pid, sorted(cov & u["mechanisms"]))
        if not best:
            return None
        pid = best[1]
        return {"paper": pid, "title": self.P[pid]["title"], "year": self.P[pid]["year"],
                "why": "Survey in the corpus covering the most of your mechanisms" + (f" ({', '.join(best[2])})" if best[2] else "") + "; skim it first to get the map."}

    # ------------------------------------------------------------------ 5. baselines & evaluation setup
    def baselines(self, u, closest, k=5):
        seen = {c["paper"] for c in closest}
        cands = []
        maxinf = max(1, max(self.P[p]["influence"] for p in self.method_ids))
        for pid in self.method_ids:
            shared = u["problems"] & set(self.P[pid]["signature"]["problems"])
            if not shared and not (u["mechanisms"] & self._prop(pid)):
                continue
            sc = sum(self._w("problem", q) for q in shared) + 0.5 * len(u["mechanisms"] & self._prop(pid)) + 2.0 * self.P[pid]["influence"] / maxinf
            cands.append((sc, pid, shared))
        cands.sort(key=lambda t: -t[0])
        out = []
        for sc, pid, shared in cands:
            if pid in seen:
                continue
            out.append({"paper": pid, "title": self.P[pid]["title"], "year": self.P[pid]["year"],
                        "why": ("addresses " + ", ".join(sorted(shared)) if shared else "proposes a mechanism you use") +
                               f"; {self.P[pid]['influence']} paper(s) in the corpus build on it"})
            if len(out) >= k:
                break
        pool = [c["paper"] for c in closest] + [b["paper"] for b in out]
        def pop(kind):
            cnt = defaultdict(int)
            for pid in pool:
                for e in self.P[pid]["signature"][kind]:
                    cnt[e] += 1
            return [{"id": e, "label": self.kg["nodes"][kind][e]["label"], "used_by": f"{n} of {len(pool)} closest/baseline papers"}
                    for e, n in sorted(cnt.items(), key=lambda kv: -kv[1])[:3]]
        return {"baselines": out, "evaluation_setup": {"benchmarks": pop("benchmarks"), "backbones": pop("models")}}

    # ------------------------------------------------------------------ 6. gaps
    def gaps(self, u):
        """
        Three kinds of gap, all restricted to *specific* problems (df < 20% of papers) so that near-universal
        problems such as 'per-task parameters' do not generate noise:
          adjacent_empty_cells : (your mechanism x problem with >=3 method papers) never applied together
          never_combined       : your mechanism + another mechanism that shares a specific problem with it,
                                 yet no paper involves both
          thin_ground          : problems you named that <=2 method papers address
        """
        spec = set(self.kg["derived"]["specific_problems"])
        out = {"adjacent_empty_cells": [], "never_combined": [], "thin_ground": []}
        n_addr = {q: len([p for p in self.PN[q]["addressed_by"] if self.P[p]["contribution"] == "METHOD"]) for q in self.PN}
        cells = []
        for m in sorted(u["mechanisms"]):
            for q, n in n_addr.items():
                if n >= 3 and (q in spec or q in u["problems"]) and q not in self.matrix.get(m, {}):
                    cells.append((n, {"mechanism": m, "problem": q,
                        "why": f"{n} method papers attack '{self.PN[q]['label']}' but none applies {self.MN[m]['label']} to it"}))
        out["adjacent_empty_cells"] = [x for _, x in sorted(cells, key=lambda t: -t[0])[:4]]
        holders = {m: {p for p in self.method_ids if m in self._mech(p)} for m in self.MN}
        mprobs = {m: {q for q in spec if q in self.matrix.get(m, {})} for m in self.MN}
        pairs = []
        for m in sorted(u["mechanisms"]):
            for m2, h2 in holders.items():
                if m2 == m or m2 in u["mechanisms"] or len(h2) < 2 or len(holders[m]) < 2 or (holders[m] & h2):
                    continue
                common = sorted(mprobs[m] & mprobs[m2])
                if common:
                    pairs.append((len(common), min(len(holders[m]), len(h2)), {"mechanism": m, "with": m2,
                        "why": f"both are applied to {', '.join(common)} ({len(holders[m])} and {len(h2)} papers), but no paper involves both"}))
        out["never_combined"] = [x for _, _, x in sorted(pairs, key=lambda t: (-t[0], -t[1]))[:4]]
        for q in sorted(u["problems"]):
            if n_addr[q] <= 2:
                out["thin_ground"].append({"problem": q, "method_papers": n_addr[q],
                                           "why": f"Only {n_addr[q]} method paper(s) in the corpus address '{self.PN[q]['label']}'"})
        return out

    # ------------------------------------------------------------------ 7. momentum
    def momentum(self, u):
        total_method = len(self.method_ids)
        recent_all = sum(1 for p in self.method_ids if self.P[p]["year"] >= 2023) / max(1, total_method)
        out = []
        for m in sorted(u["mechanisms"] | u["foils"]):
            per = self.MN[m]["proposals_by_period"]
            tot = sum(per.values())
            share = per[">=2023"] / tot if tot else 0
            if tot < 3:
                trend = "too few proposals in the corpus to call a trend"
            elif share > recent_all + 0.15:
                trend = "rising: a larger share of its proposals are from 2023+ than the corpus average"
            elif share < recent_all - 0.15:
                trend = "settled: most proposals are older than the corpus average"
            else:
                trend = "steady"
            out.append({"mechanism": m, "label": self.MN[m]["label"], "proposals_by_period": per, "trend": trend})
        return out

    # ------------------------------------------------------------------ orchestrate
    def run(self, text, level="beginner", k=5, max_papers=12):
        u = self.understand(text)
        closest = self.closest_work(u, k)
        nov = self.novelty(u)
        path = self.reading_path(u, closest, level, max_papers) if closest else {"level": level, "steps": [], "orientation": None}
        bl = self.baselines(u, closest)
        gp = self.gaps(u)
        mom = self.momentum(u)
        landscape = [{"problem": q, "label": self.PN[q]["label"],
                      "mechanisms_applied": {m: c for m, c in list(self.PN[q]["mechanisms_applied"].items())[:5]},
                      "papers": len(self.PN[q]["addressed_by"])} for q in sorted(u["problems"])]
        recurring = defaultdict(list)
        for c in closest:
            for a in self.P[c["paper"]]["authors"]:
                recurring[a].append(self.short(c["paper"]))
        queries = []
        ml = [self.MN[m]["label"].split("(")[0].strip().lower() for m in sorted(u["mechanisms"])]
        pl = [self.PN[q]["label"].lower() for q in sorted(u["problems"])]
        if ml:
            queries.append(" ".join(ml))
            if pl:
                queries.append(" ".join(ml[:2]) + " " + pl[0])
        queries += [self.P[c["paper"]]["title"] + " citations" for c in closest[:1]]
        caveats = [f"Knowledge is limited to the {len(self.P)} papers in the corpus; 'novel' means 'not in this corpus'. Verify with an external search (queries suggested).",
                   "Entities were mapped by a hand-written lexicon: unusual phrasings of an idea may be missed (see 'understanding')."]
        if any("unverified" in s for s in self.kg["meta"]["reference_sources"]):
            caveats.append("This knowledge state was built from the bootstrap corpus whose citation edges are unverified; rebuild from Semantic Scholar data (fetch_papers.py) before relying on it.")
        if u["confidence"] == "low":
            caveats.append("Few entities were recognised in the input; add concrete mechanisms (e.g. 'low-rank', 'quantization') and problems (e.g. 'memory', 'forgetting') for a sharper result.")
        return {
            "input": text,
            "understanding": {"confidence": u["confidence"],
                              "mechanisms": {m: {"label": self.MN[m]["label"], "family": self.MN[m]["family"], "matched": u["raw"]["mechanisms"][m]["match"]} for m in sorted(u["mechanisms"])},
                              "baselines_or_foils_you_mention": sorted(u["foils"]),
                              "problems": {q: {"label": self.PN[q]["label"], "matched": u["raw"]["problems"][q]["match"]} for q in sorted(u["problems"])},
                              "concepts": sorted(u["concepts"]), "models": sorted(u["models"]), "benchmarks": sorted(u["benchmarks"])},
            "novelty_assessment": nov,
            "closest_prior_work": closest,
            "reading_path": path,
            "baselines_and_evaluation": bl,
            "open_gaps": gp,
            "field_momentum": mom,
            "problem_landscape": landscape,
            "recurring_lead_authors": {a: v for a, v in recurring.items() if len(v) > 1},
            "queries_to_verify_externally": queries,
            "caveats": caveats,
        }

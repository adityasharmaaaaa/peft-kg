"""Render the reasoner's structured output as a readable markdown report."""

VERDICTS = {
    "ALREADY_COVERED": "Likely already done - a corpus paper contains your mechanisms and targets your problems.",
    "INCREMENTAL": "Incremental - closest paper covers your mechanisms and part of your problem set.",
    "NOVEL_APPLICATION": "Novel application - the mechanisms exist, but nobody applied them to your problem.",
    "NOVEL_COMBINATION": "Novel combination - each piece exists, no paper combines them.",
    "MECHANISM_SET_EXISTS": "Mechanism set exists in the corpus; no problem named so the angle cannot be judged.",
    "MECHANISM_NOT_IN_CORPUS": "The corpus has no method paper for a mechanism you mention - cannot judge.",
    "NOT_ASSESSED": "No mechanism recognised - novelty not assessed.",
}


def render(out):
    L, w = [], lambda s="": L.append(s)
    u, n = out["understanding"], out["novelty_assessment"]
    w("# PEFT onboarding report\n")
    w(f"> {out['input'].strip()}\n")
    w("## 1. How the system understood your input\n")
    w(f"Confidence: **{u['confidence']}**")
    w("- Mechanisms: " + (", ".join(f"{v['label']} [`{k}`, matched \"{v['matched']}\"]" for k, v in u["mechanisms"].items()) or "none recognised"))
    if u["baselines_or_foils_you_mention"]:
        w("- Treated as baselines / foils (appear only in contrastive sentences): " + ", ".join(u["baselines_or_foils_you_mention"]))
    w("- Problems: " + (", ".join(f"{v['label']} [`{k}`]" for k, v in u["problems"].items()) or "none recognised"))
    if u["models"] or u["benchmarks"] or u["concepts"]:
        w("- Also recognised: " + "; ".join(x for x in (
            "models " + ", ".join(u["models"]) if u["models"] else "", "benchmarks " + ", ".join(u["benchmarks"]) if u["benchmarks"] else "",
            "concepts " + ", ".join(u["concepts"]) if u["concepts"] else "") if x))
    w("\n## 2. Novelty assessment\n")
    w(f"**{n['verdict']}** - {VERDICTS.get(n['verdict'], '')}\n")
    for r in n.get("rationale", []) + ([n["explanation"]] if "explanation" in n else []):
        w(f"- {r}")
    for c in n.get("untested_cells", []):
        w(f"- Untested pairing: `{c['mechanism']}` x `{c['problem']}` - {c['reason']}")
    w("\n## 3. " + ("Closest prior work (read these to see if you are scooped)" if n["verdict"] != "NOT_ASSESSED"
                    else "Papers attacking the problems you named (no mechanism given, so this is a map, not a scoop-check)") + "\n")
    if not out["closest_prior_work"]:
        w("Nothing in the corpus overlaps with the recognised entities.")
    for c in out["closest_prior_work"]:
        w(f"- **{c['title']}** ({c['year']}) - score {c['score']}. {c['relation_to_your_idea'].capitalize()}.")
        bits = []
        if c["shared_mechanisms"]: bits.append("shares " + ", ".join(c["shared_mechanisms"]))
        if c["shared_problems"]: bits.append("same problems: " + ", ".join(c["shared_problems"]))
        if c["your_mechanisms_it_lacks"]: bits.append("lacks " + ", ".join(c["your_mechanisms_it_lacks"]))
        if c["its_mechanisms_you_lack"]: bits.append("also has " + ", ".join(c["its_mechanisms_you_lack"]))
        if bits: w("  - " + "; ".join(bits))
    rp = out["reading_path"]
    w(f"\n## 4. Reading path ({rp['level']})\n")
    if rp.get("orientation"):
        o = rp["orientation"]
        w(f"**Step 0 - orientation:** {o['title']} ({o['year']}). {o['why']}\n")
    stage = None
    for s in rp["steps"]:
        if s["stage"] != stage:
            stage = s["stage"]; w(f"**{stage}**")
        after = f" (after: {', '.join(s['read_after'])})" if s["read_after"] else ""
        w(f"{s['step']}. {s['title']} ({s['year']}){after}  \n   _{s['why']}_")
    b = out["baselines_and_evaluation"]
    w("\n## 5. Baselines and evaluation setup\n")
    for x in b["baselines"]:
        w(f"- Compare against **{x['title']}** ({x['year']}): {x['why']}.")
    es = b["evaluation_setup"]
    if es["benchmarks"]: w("- Common benchmarks in this neighbourhood: " + "; ".join(f"{x['label']} ({x['used_by']})" for x in es["benchmarks"]))
    if es["backbones"]: w("- Common backbones: " + "; ".join(f"{x['label']} ({x['used_by']})" for x in es["backbones"]))
    g = out["open_gaps"]
    w("\n## 6. Open gaps near your idea\n")
    for x in g["adjacent_empty_cells"]: w(f"- Empty cell: `{x['mechanism']}` x `{x['problem']}` - {x['why']}")
    for x in g["never_combined"]: w(f"- Never combined: `{x['mechanism']}` + `{x['with']}` - {x['why']}")
    for x in g["thin_ground"]: w(f"- Thin ground: `{x['problem']}` - {x['why']}")
    if not any(g.values()): w("- None found.")
    w("\n## 7. Field momentum\n")
    for x in out["field_momentum"]:
        w(f"- {x['label']}: proposals by period {x['proposals_by_period']} - {x['trend']}")
    if out["problem_landscape"]:
        w("\n## 8. Problem landscape\n")
        for x in out["problem_landscape"]:
            w(f"- {x['label']}: {x['papers']} papers; mechanisms most applied: " + ", ".join(f"{m} ({c})" for m, c in x["mechanisms_applied"].items()))
    if out["recurring_lead_authors"]:
        w("\n## 9. Recurring lead authors in the closest work\n")
        for a, v in out["recurring_lead_authors"].items(): w(f"- {a}: {', '.join(v)}")
    w("\n## Verify externally\n")
    for q in out["queries_to_verify_externally"]: w(f"- `{q}`")
    w("\n## Caveats\n")
    for c in out["caveats"]: w(f"- {c}")
    return "\n".join(L)

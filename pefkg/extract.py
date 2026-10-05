"""
Rule-based mapping from text to ontology entities.

No NER / relation-extraction library is used. The only machinery is `re`. All the
judgement lives in two places: the lexicon in `ontology.py` and the role rules below.

ROLE RULES (papers)
-------------------
After roles are assigned, a paper that PROPOSES a mechanism in ontology.OPERATES_OVER (composition,
routing, unification, serving) has its other PROPOSES demoted to USES: it operates over them.

For each mechanism that has a *core* hit anywhere in title + summary:

  PROPOSES      core hit in the title, or in the first "propose / introduce / present ..."
                sentence - but only if the paper has such a sentence at all and is not a
                survey. (Analysis papers that study LoRA do not propose LoRA.)
  COVERS        (surveys only) every mechanism a survey discusses.
  USES          any other hit in a non-contrastive sentence, or a *derived* hit (a named
                variant such as QLoRA / AdaLoRA / S-LoRA) anywhere.
  COMPARES_WITH hits that occur only in contrastive sentences (unlike / than / without ...).

Concepts: ESTABLISHES if hit in the title or a claim sentence, else RELIES_ON.
Problems: any hit counts as ADDRESSES (motivation or measured outcome).
Contribution type: SURVEY > BENCHMARK > METHOD > FOUNDATION_MODEL > ANALYSIS > OTHER.
"""
import re
from . import ontology as O

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(0-9])")


def normalise(text: str) -> str:
    text = (text or "").replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str):
    return [s.strip() for s in _SENT_SPLIT.split(normalise(text)) if s.strip()]


def _hits(patterns, text_lc):
    out = []
    for p in patterns:
        for m in re.finditer(p, text_lc):
            out.append(m.group(0))
    return out


_PROPOSE_RE = re.compile(r"\b" + O.PROPOSE_VERBS)
_PROPOSE_START_RE = re.compile(O.PROPOSE_VERBS_START)
_OBSERVE_RE = re.compile(r"\b" + O.OBSERVE_VERBS)
_CONTRAST_RE = re.compile(O.CONTRAST_CUES)
_SURVEY_RE = re.compile(O.SURVEY_CUES)
_MODEL_CUE_RE = re.compile(r"\b(model|models|architecture|network|framework)\b")


def _is_propose_sentence(s_lc: str) -> bool:
    return bool(_PROPOSE_RE.search(s_lc) or _PROPOSE_START_RE.match(s_lc))


def _entity_hits(table, text_lc, key="patterns"):
    res = {}
    for eid, spec in table.items():
        h = _hits(spec[key], text_lc)
        if h:
            res[eid] = h
    return res


def analyze_paper(title: str, summary: str) -> dict:
    title_lc = normalise(title).lower()
    sents = split_sentences(summary)
    sents_lc = [s.lower() for s in sents]

    # ---- locate claim sentences ------------------------------------------------------
    propose_idx = next((i for i, s in enumerate(sents_lc) if _is_propose_sentence(s)), None)
    observe_idx = next((i for i, s in enumerate(sents_lc) if _OBSERVE_RE.search(s)), None)
    claim_idx = {i for i in (propose_idx, observe_idx, 0 if sents else None) if i is not None}

    first_two = " ".join([title_lc] + sents_lc[:2])
    is_survey = bool(_SURVEY_RE.search(first_two))

    # ---- mechanisms ------------------------------------------------------------------
    mechanisms = {}
    for mid, spec in O.MECHANISMS.items():
        core = spec["core"]
        derived = spec.get("derived", [])
        title_core = _hits(core, title_lc)
        prop_core = _hits(core, sents_lc[propose_idx]) if propose_idx is not None else []
        # mentions per sentence
        plain, contrast = [], []
        for s in sents_lc:
            hs = _hits(core, s) + _hits(derived, s)
            if hs:
                (contrast if _CONTRAST_RE.search(s) else plain).extend(hs)
        title_derived = _hits(derived, title_lc)

        role, evidence = None, None
        proposing_paper = propose_idx is not None and not is_survey
        if is_survey and (title_core or plain or contrast or prop_core or title_derived):
            role, evidence = "COVERS", ("survey", (title_core or plain or contrast or prop_core or title_derived)[0])
        elif proposing_paper and title_core:
            role, evidence = "PROPOSES", ("title", title_core[0])
        elif proposing_paper and prop_core:
            role, evidence = "PROPOSES", ("propose_sentence", prop_core[0])
        elif plain or title_core or title_derived or prop_core:
            hit = (plain or title_core or title_derived or prop_core)[0]
            role, evidence = "USES", ("mention", hit)
        elif contrast:
            role, evidence = "COMPARES_WITH", ("contrastive_mention", contrast[0])
        if role:
            mechanisms[mid] = {"role": role, "evidence_type": evidence[0], "match": evidence[1]}

    # ---- composition / unification papers do not own their constituents ---------------
    owners = [m for m, v in mechanisms.items() if v["role"] == "PROPOSES" and m in O.OPERATES_OVER]
    if owners:
        for m, v in mechanisms.items():
            if v["role"] == "PROPOSES" and m not in O.OPERATES_OVER:
                v["role"], v["evidence_type"] = "USES", "constituent_of:" + owners[0]

    # ---- problems --------------------------------------------------------------------
    full_lc = " ".join([title_lc] + sents_lc)
    problems = {pid: {"match": h[0], "mentions": len(h)} for pid, h in _entity_hits(O.PROBLEMS, full_lc).items()}

    # ---- concepts --------------------------------------------------------------------
    concepts = {}
    claim_text = " ".join(sents_lc[i] for i in sorted(claim_idx))
    for cid, h in _entity_hits(O.CONCEPTS, full_lc).items():
        in_claim = bool(_hits(O.CONCEPTS[cid]["patterns"], title_lc + " " + claim_text))
        concepts[cid] = {"role": "ESTABLISHES" if in_claim else "RELIES_ON", "match": h[0]}

    # ---- models / benchmarks ---------------------------------------------------------
    models = {mid: {"match": h[0], "in_title": bool(_hits(O.MODELS[mid]["patterns"], title_lc)),
                    "in_propose_sentence": bool(propose_idx is not None and _hits(O.MODELS[mid]["patterns"], sents_lc[propose_idx]))}
              for mid, h in _entity_hits(O.MODELS, full_lc).items()}
    benchmarks = {bid: {"match": h[0], "in_title": bool(_hits(O.BENCHMARKS[bid]["patterns"], title_lc))}
                  for bid, h in _entity_hits(O.BENCHMARKS, full_lc).items()}

    # ---- contribution type -----------------------------------------------------------
    proposes = [m for m, v in mechanisms.items() if v["role"] == "PROPOSES"]
    if is_survey:
        contribution = "SURVEY"
    elif any(b["in_title"] for b in benchmarks.values()) and not proposes:
        contribution = "BENCHMARK"
    elif proposes:
        contribution = "METHOD"
    elif propose_idx is not None and _MODEL_CUE_RE.search(sents_lc[propose_idx]):
        contribution = "FOUNDATION_MODEL"
    elif any(c["role"] == "ESTABLISHES" and not O.CONCEPTS[k].get("generic") for k, c in concepts.items()) or observe_idx is not None:
        contribution = "ANALYSIS"
    else:
        contribution = "OTHER"

    return {
        "contribution": contribution,
        "mechanisms": mechanisms,
        "problems": problems,
        "concepts": concepts,
        "models": models,
        "benchmarks": benchmarks,
        "claim_sentence": sents[propose_idx] if propose_idx is not None else (sents[0] if sents else ""),
    }


# --------------------------------------------------------------------------------------
# New input (a researcher's idea / situation described in free text)
# --------------------------------------------------------------------------------------
def _foil_spans(s_lc):
    """Character spans that a contrast cue scopes over: from the cue to the next comma/semicolon/period."""
    spans = []
    for m in _CONTRAST_RE.finditer(s_lc):
        end = len(s_lc)
        for ch in (",", ";", "."):
            i = s_lc.find(ch, m.end())
            if i != -1:
                end = min(end, i)
        spans.append((m.start(), end))
    return spans


def analyze_input(text: str) -> dict:
    """
    Map a free-text description of an idea or situation onto the same ontology.

    Contrast is *scoped*: in "Unlike QLoRA, the rank budget is not fixed" only QLoRA (the clause the
    cue governs) is a baseline/foil; "rank budget" is still part of the idea. (Paper abstracts use the
    simpler sentence-level rule because they are short and single-clause; user input is not.)
    """
    sents = split_sentences(text)
    sents_lc = [s.lower() for s in sents]
    full_lc = " ".join(sents_lc)

    mechanisms = {}
    for mid, spec in O.MECHANISMS.items():
        pats = spec["core"] + spec.get("derived", [])
        plain, contrast = [], []
        for s in sents_lc:
            spans = _foil_spans(s)
            for p in pats:
                for m in re.finditer(p, s):
                    (contrast if any(a <= m.start() < b for a, b in spans) else plain).append(m.group(0))
        if plain:
            mechanisms[mid] = {"role": "IN_IDEA", "match": plain[0]}
        elif contrast:
            mechanisms[mid] = {"role": "BASELINE_OR_FOIL", "match": contrast[0]}
    return {
        "mechanisms": mechanisms,
        "problems": {pid: {"match": h[0]} for pid, h in _entity_hits(O.PROBLEMS, full_lc).items()},
        "concepts": {cid: {"match": h[0]} for cid, h in _entity_hits(O.CONCEPTS, full_lc).items()},
        "models": {mid: {"match": h[0]} for mid, h in _entity_hits(O.MODELS, full_lc).items()},
        "benchmarks": {bid: {"match": h[0]} for bid, h in _entity_hits(O.BENCHMARKS, full_lc).items()},
    }

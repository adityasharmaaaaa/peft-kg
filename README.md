# PEFT Onboarding Knowledge Graph

A system that turns a small corpus of research papers on **parameter-efficient fine-tuning (PEFT) of language models** into a structured knowledge graph (typed entities + typed relations), then reasons over it when a researcher describes a **new idea or situation**, and returns an actionable report: *is it novel, what is the closest prior work, what to read and in what order, what to compare against, where the gaps are.*

> **Start with [`approach.md`](approach.md)** (design reasoning, tradeoffs, limitations).
>
> **Data provenance warning.** The bundled `data/papers_raw.json` is a **hand-curated bootstrap** (paraphrased summaries, unverified citation edges) because the build environment could not reach Semantic Scholar. Run `python fetch_papers.py` to replace it with real API data, then rebuild. Details in `approach.md`.

## Requirements

* Python 3.9+ (developed on 3.12). **No third-party packages**; standard library only.
* Network access only for `fetch_papers.py`. Everything else works offline.
* Optional: `S2_API_KEY` environment variable (free key from semanticscholar.org/product/api) for a higher Semantic Scholar rate limit.

```
git clone https://github.com/adityasharmaaaaa/peft-kg.git && cd peft-kg      # nothing to pip install
```

## Quick start (offline, bundled data)

```bash
python -m pefkg build                                   # raw data -> knowledge_state/kg.json (+ kg_summary.md)
python -m pefkg ask "I want LoRA on a 4-bit quantized LLaMA with the rank allocated adaptively per layer, to cut GPU memory. Unlike QLoRA the rank budget is not fixed."
```

The knowledge state is **already committed** in `knowledge_state/`, so `ask` works without running `build`.

## Testable interface

```bash
python -m pefkg ask "<free text: an idea, a draft abstract, or a situation>"          # markdown report
python -m pefkg ask "<text>" --level expert          # shorter path: skips generic foundation papers
python -m pefkg ask "<text>" --json                  # structured output (this is the primary output format)
python -m pefkg ask --file idea.txt
echo "<text>" | python -m pefkg ask
python -m pefkg inspect lora                         # a node + every edge touching it (papers, mechanisms, problems ...)
python -m pefkg inspect low_rank_update
```

Try these (they exercise different verdicts):

| input | expected verdict |
|---|---|
| `We freeze pretrained weights and inject trainable low-rank decomposition matrices so GPT-3 fine-tuning needs less GPU memory and adds no inference latency.` | `ALREADY_COVERED` (LoRA, identified as originator) |
| `4-bit quantized LLaMA with LoRA where the rank is allocated adaptively per layer, to cut GPU memory. Unlike QLoRA the rank budget is not fixed.` | `NOVEL_COMBINATION` |
| `Use prefix tuning with soft prompts to handle long-context adaptation of sequence length.` | `NOVEL_APPLICATION` |
| `I'm new and want cheaper fine-tuning memory without forgetting.` | `NOT_ASSESSED` + a reading path |

Output sections: how the input was understood (with matched terms) -> novelty verdict with rationale -> closest prior work -> reading path (ordered, each step justified) -> baselines & evaluation setup -> open gaps -> field momentum -> queries to verify externally -> caveats.

## Loading / regenerating the knowledge state

| goal | command |
|---|---|
| Rebuild from the bundled raw data | `python -m pefkg build` |
| Use **real** Semantic Scholar data | `python fetch_papers.py` (optionally `S2_API_KEY=... python fetch_papers.py`), then `python -m pefkg build` |
| Rebuild the bundled bootstrap raw data | `python data/bootstrap_corpus.py` |
| Build from another file / location | `python -m pefkg build --raw path/papers_raw.json --out some_dir` and `python -m pefkg ask "..." --kg some_dir/kg.json` |

`fetch_papers.py` resolves each title in `data/seed_titles.tsv` via the Semantic Scholar `search/match` endpoint, pulls metadata and reference lists, keeps only references that are themselves seeds (in-corpus citation edges), and caches responses in `data/s2_cache.json` (resumable). To change topic, replace the seed list and the lexicon in `pefkg/ontology.py`.

## Inspecting the knowledge state without running code

| file | what it is |
|---|---|
| `knowledge_state/kg.json` | The serialized graph. `schema` (entity types, 22 relation types with semantics, mechanism taxonomy), `nodes` (papers with extracted signatures, mechanisms, problems, concepts, models, benchmarks, authors), `edges` (flat typed list; **every edge has an `evidence` string**), `derived` (IDF, mechanism x problem coverage matrix). |
| `knowledge_state/kg_summary.md` | Human-readable view: edge counts, mechanism taxonomy with who proposed what and when, papers by tier, a Mermaid lineage graph, the coverage matrix. |

Edge format: `{"source": "paper:adalora", "relation": "BUILDS_ON", "target": "paper:lora", "evidence": "A uses low_rank_update that B proposes"}`.

## Tests

```bash
python -m unittest tests.test_pipeline      # 23 tests, <1 s, offline
```

## Project layout

```
approach.md                 design decisions, tradeoffs, limitations, next steps  (read first)
README.md                   this file
fetch_papers.py             Semantic Scholar -> data/papers_raw.json (stdlib only)
data/
  bootstrap_corpus.py       hand-curated 65-paper bootstrap (generates the two files below)
  papers_raw.json           raw records: title, year, lead author, abstract/summary, in-corpus references
  seed_titles.tsv           seed list for fetch_papers.py
pefkg/
  ontology.py               entity types, 22 relation types, mechanism taxonomy, lexicon (all hand-written)
  extract.py                text -> entities with roles (PROPOSES / USES / COMPARES_WITH / COVERS ...)
  build.py                  typed citation rules, derived structure, serialisation
  summary.py                markdown export of the knowledge state
  reason.py                 novelty verdict, closest work, reading path, baselines, gaps, momentum
  render.py                 report formatting
  __main__.py               CLI (build / ask / inspect)
knowledge_state/            kg.json, kg_summary.md
tests/test_pipeline.py
```

## Configuration

| variable / flag | meaning |
|---|---|
| `S2_API_KEY` | optional; Semantic Scholar API key used by `fetch_papers.py` |
| `--level beginner\|expert` | reading-path depth (beginner includes foundation papers) |
| `--max-papers N` | cap on reading-path length (default 12) |
| `--kg PATH` | knowledge state to reason over (default `knowledge_state/kg.json`) |

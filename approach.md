# Approach

**Domain:** Research Paper Onboarding  **Topic:** parameter-efficient fine-tuning (PEFT) of language models
**Stack:** Python standard library only. No NER / relation-extraction / graph-construction tool or LLM is used anywhere.

> ## Read this first: data provenance
> The environment this was built in could not reach `api.semanticscholar.org` or arXiv (HTTP 403 from the sandbox), so I could **not** download the dataset myself.
> What I did instead:
> * `fetch_papers.py` implements the real Semantic Scholar pipeline (title match, metadata, reference lists restricted to the corpus, arXiv-year normalisation). Its parsing functions are unit-tested against mocked API payloads, but **it has never been run against the live API**.
> * The bundled snapshot (`data/papers_raw.json`) is a **bootstrap corpus I wrote from my knowledge of the literature**: titles, years and lead authors are from memory; each "abstract" is a **short paraphrase I wrote**, not the published abstract; the 435 in-corpus citation edges are **best-effort and unverified**.
> * Every node and the knowledge state record this (`abstract_source: hand_written_paraphrase`, `reference_source: curated_from_memory_unverified`), and every report prints a caveat while that is the case.
> * **Before relying on any output, run `python fetch_papers.py && python -m pefkg build`.** The schema, extractor, graph builder and reasoner are unchanged by that swap; only the facts underneath get real. Expect the extractor to be somewhat less accurate on real abstracts than on my paraphrases, which I wrote in a style that suits the lexicon (see Limitations).
> * The corpus stops at mid-2024.

---

## 1. The problem I chose

A researcher enters a subfield with a half-formed idea. Two questions decide whether they waste months:

1. **"Is this already done, and by whom?"** (novelty / closest prior work)
2. **"What do I have to read, in what order, to be able to work on it?"** (a prerequisite path *specific to the idea*, not a generic reading list)

The citation network contains the answer to both, but only if you know *why* one paper cites another. "A cites B" can mean "A is built on B", "A beats B", "A assumes a fact B proved", or "A mentions B in passing". Those lead to opposite reading advice. So the core of this project is a schema that classifies citations and papers structurally, so that a new idea can be placed in the graph and reasoned about.

I picked PEFT because (a) it moved fast (2019-2024) with a clear lineage and many near-duplicate-sounding papers, which is exactly where "is this novel?" is hard; (b) it has a natural mechanism x problem structure; (c) I know the literature well enough to hand-label a gold set and sanity-check behaviour.

## 2. Subset of data and why

65 papers in three layers:

| layer | papers | why included |
|---|---|---|
| Foundations (Transformer, ULMFiT, BERT, GPT-2/3, T5, LLaMA, Llama 2, GLUE, SuperGLUE, MMLU, scaling laws, Chinchilla, FLAN, InstructGPT, Self-Instruct, LIMA ...) | 18 | A newcomer's path has to bottom out somewhere; these are the shared substrate and also supply backbones/benchmarks |
| PEFT lineage 2019-2024 (Adapters -> Prefix/Prompt tuning -> BitFit -> LoRA -> unified views -> AdaLoRA/DoRA/VeRA/...; surveys) | 35 | The topic itself |
| Efficiency stack that PEFT is routinely combined with (quantization, mixed precision, checkpointing, 8-bit optimizers, GPTQ/AWQ, QLoRA/LoftQ/QA-LoRA, EWC, task arithmetic) | 12 | Many "new" PEFT ideas are combinations with this layer; without it the novelty question is meaningless |

I deliberately did **not** go broad (no vision/diffusion PEFT, no RAG, no pretraining-efficiency work): the brief says relationships inside a well-chosen subset matter more than coverage, and a boundary I can name makes "novel" mean something: *not in this corpus*. The reasoner says so in every report.

Result: 65 papers, 435 in-corpus citation edges, 31 mechanisms in 8 families, 19 problems, 8 concepts, 6 model families, 6 benchmarks, 57 lead authors; 853 edges in the final graph.

## 3. Knowledge model

### 3.1 The central decision: a paper is a *signature*, not text

A paper is represented by what it does along five orthogonal dimensions, each a controlled vocabulary I defined (`pefkg/ontology.py`):

| entity | what it captures | why it exists |
|---|---|---|
| **Mechanism** (+ **MechanismFamily**) | the technical move: low-rank update, soft prompt, 4-bit quantization, adaptive rank ... | The unit of novelty. Two papers are "the same idea" when their mechanisms match |
| **Problem** | obstacle attacked: training memory, forgetting, rank selection, serving ... | Novelty is relative to a *purpose*; the same mechanism on a new problem is a different contribution |
| **Concept** | idea a paper establishes or rests on (intrinsic dimensionality, scaling laws ...) | Explains *motivation* edges that are not method-to-method |
| **Model** / **Benchmark** | backbone family; evaluation suites | Practical output: what to evaluate on; also separates "foundation" papers |
| **Author** (lead only) | who recurs | Cheap answer to "who works on this" |

Why not a flat document store / embeddings: you cannot ask a vector index "has anyone applied mechanism M to problem P?" or "which mechanisms have never been combined?". Those questions are queries over a *matrix of typed relations*, which is what the knowledge state is (`derived.mechanism_problem_matrix`: 94 of 589 mechanism x problem cells are populated).

**Mechanism taxonomy (2 levels).** Families describe *where in the stack the method acts*: ADDITIVE, SELECTIVE, REPARAMETERIZATION, COMPRESSION, COMPOSITION, TRAINING_RECIPE_AND_SYSTEMS, SUPERVISION, HYBRID. I started from the additive/selective/reparameterization split used in the PEFT literature and added COMPRESSION, COMPOSITION, SYSTEMS and SUPERVISION because the corpus's second layer (QLoRA, LoRAHub, S-LoRA, LIMA) does not fit the standard three. Leaf mechanisms are intentionally fine-grained inside REPARAMETERIZATION (adaptive rank, weight decomposition, informed init, parameter sharing, LR asymmetry, frozen projection ...) because that is where the near-duplicate papers live: a coarse "low-rank" node would collapse DoRA, PiSSA, VeRA and LoRA+ into one, and the novelty check would be useless.

### 3.2 Roles: one mention, three meanings

A sentence that names a mechanism can mean the paper **introduces** it, **uses** it as substrate, or **argues against** it. Collapsing these is the most common reason a text-derived graph gives wrong reading advice, so the schema keeps them apart:

| relation | rule (all in `extract.py`) |
|---|---|
| `PROPOSES` | core hit in the **title** or in the **first "propose / introduce / present ..." sentence**; paper must have such a sentence and not be a survey |
| `USES` | other non-contrastive mention, or a *named variant* (QLoRA, AdaLoRA, S-LoRA...) which tells us the paper builds on LoRA, not that it introduces it |
| `COMPARES_WITH` | mention only inside a contrastive sentence (unlike / than / without / outperforms ...) |
| `COVERS` | surveys only |

Two refinements came from failures I saw when I printed the extraction for every paper:
* **`OPERATES_OVER` rule.** Composition/unification/serving papers (AdapterFusion, UniPELT, LoRAHub, SPoT, S-LoRA, AdaMix) name the mechanisms they combine in their first sentence, so they were wrongly credited with *proposing* adapters / LoRA / prompts. Rule: a paper that proposes a mechanism in `OPERATES_OVER` has its other proposals demoted to `USES`.
* **Derived-name patterns.** `\w+-?lora` variants are `derived`, never `core`, so "QLoRA" in a title does not make QLoRA the proposer of low-rank updates.

### 3.3 Typed citations: 435 raw citations -> 8 relation types

Raw citation data only says that A cites B. `build.type_citation` decides *why it matters*, using the signatures of both papers, in this priority order:

| type | rule | use downstream |
|---|---|---|
| `SURVEYS` | A is a survey | never a prerequisite |
| `USES_BENCHMARK` / `USES_BACKBONE` / `BACKGROUND` | B is a benchmark or foundation-model paper; checks whether A actually evaluates on it / adapts it | backbones only for beginners |
| `EXTENDS` | A and B **propose a shared mechanism** | strong prerequisite |
| `BUILDS_ON` | A **uses** a mechanism B proposes | strong prerequisite |
| `COMPARES_WITH_PAPER` | A contrasts with B's mechanism, or attacks the same *specific* problem with a different mechanism | "must compare", **not** a prerequisite |
| `MOTIVATED_BY` | A rests on a (non-generic) concept B establishes, or on B's analysis of a mechanism A uses | prerequisite |
| `BACKGROUND` | none of the above | ignored for expert paths |

Distribution in the bundled graph: BACKGROUND 191, SURVEYS 79, EXTENDS 46, COMPARES_WITH_PAPER 34, USES_BACKBONE 31, BUILDS_ON 30, USES_BENCHMARK 16, MOTIVATED_BY 8. About 44% of raw citations carry no structural meaning at all, and survey citations (18%) would otherwise swamp every reading path. That is the practical value of typing them.

**Generic vs specific.** Concepts like "pre-train then fine-tune" and problems like "per-task parameters" appear in nearly every paper and would turn every citation into a "shared problem". Using document frequency over the corpus, concepts/problems above a threshold (25% / 20% of papers) are not used to create `MOTIVATED_BY` / alternative-approach edges or gap candidates, and all overlap scoring is IDF-weighted.

### 3.4 Derived structure stored in the knowledge state

* `influence` per paper: number of papers that transitively require it (over prerequisite edges). Foundation-vs-frontier signal that raw citation counts (unavailable offline) would otherwise supply.
* `tier`: FOUNDATION / ANALYSIS / CORE_METHOD (first in corpus to propose some mechanism) / EXTENSION / SURVEY.
* per-mechanism `first_in_corpus`, proposers in order, and proposals by period (<=2020 / 2021-22 / >=2023), which is the data behind "field momentum".
* `mechanism_problem_matrix` with the supporting paper ids in every cell.
* `idf` weights and the generic/specific split.
* `UNDERPINS` (Concept -> Mechanism): small hand-authored background knowledge (e.g. intrinsic dimensionality underpins low-rank updates). Used for explanation, not scoring.

## 4. How the knowledge representation is built

```
data/papers_raw.json --extract.analyze_paper--> signatures (mechanism roles, problems, concepts, models, benchmarks, contribution type)
                     --build.type_citation----> typed paper->paper edges
                     --derived structure------> influence, tiers, originators, coverage matrix, IDF
                     ------------------------->  knowledge_state/kg.json  (+ kg_summary.md)
```

Contribution type (SURVEY > BENCHMARK > METHOD > FOUNDATION_MODEL > ANALYSIS) is itself rule-derived and drives edge typing: that is how "BERT" gets treated as foundation and "Intrinsic Dimensionality ..." as analysis without any per-paper annotation.

**Tradeoffs I made on purpose**
* *Hand-written lexicon vs. statistical extraction.* The brief requires my own entity definitions and mapping logic, and a regex lexicon makes every edge explainable (`evidence` on every edge: `title: "lora"`, `both propose quantization`). The cost is recall on unusual phrasing and a lexicon that must grow with the field.
* *Sentence-level cues vs. parsing.* "First propose sentence" and "contrastive sentence" are crude proxies for what a paper claims. They are cheap, inspectable, and measurable (below); a dependency parse would be better and I did not add one.
* *Small and typed vs. large and shallow.* 65 papers means I can hand-check roles; it also means the system can only say "not in this corpus".
* *Single-label roles.* A paper has one role per mechanism; reality has gradations (QLoRA both uses LoRA and modifies how it is trained).

**Measured extraction quality** (`tests/test_pipeline.py`): on 18 papers whose PROPOSES sets I hand-labelled from my knowledge of the papers: **precision 0.87, recall 1.00** (20 TP, 3 FP). The 3 FPs share one cause: a LoRA-variant's title names LoRA (LoRA+, DoRA, LongLoRA), so it is credited with proposing low-rank updates. Caveat: I looked at these papers while tuning the rules, so this is a regression check, **not a held-out estimate**; it is also measured on my paraphrased summaries, not real abstracts.

## 5. What happens when a new input arrives

Input: any free text describing an idea or situation. `python -m pefkg ask "..."`. The reasoner loads **only the serialized `kg.json`** and the lexicon.

1. **Understand.** The same lexicon maps the text to mechanisms, problems, concepts, models. Contrast is *scoped to the clause after the cue*: in "Unlike QLoRA, the rank budget is not fixed", QLoRA is a foil, "rank budget" stays in the idea (an early version demoted the whole sentence and silently dropped the user's own idea; there is a test for it). Confidence is reported from how many entities were recognised.
2. **Closest prior work.** For each method/analysis paper: IDF-weighted overlap on mechanisms (weight 0.55), problems (0.30), concepts+models (0.15), averaging Jaccard (similarity) and containment (how much of *your* idea the paper already contains). Rewards rare shared mechanisms; each result lists what it shares, lacks, and adds.
3. **Novelty verdict**, a rule-based decision from the coverage structure, not a score:

   | verdict | condition |
   |---|---|
   | `ALREADY_COVERED` | one method paper involves all your mechanisms **and** addresses all your problems (also says if that paper is the *originator*, so descendants are not mistaken for prior art) |
   | `INCREMENTAL` | such a paper covers the mechanisms and part of the problems |
   | `NOVEL_APPLICATION` | the mechanism set exists but no paper targets your problems |
   | `NOVEL_COMBINATION` | every mechanism has holders but no single paper holds them all; lists the maximal covered sub-combinations |
   | `MECHANISM_NOT_IN_CORPUS` / `NOT_ASSESSED` | the corpus can't judge (said explicitly) |

4. **Reading path.** Roots = top-3 closest papers + the originator of each mechanism in the idea. Walk *prerequisite* edges backwards (EXTENDS / BUILDS_ON / MOTIVATED_BY; plus backbone/foundation edges for beginners), depth-limited (3 beginner / 2 expert), drop surveys (a survey is orientation, not a prerequisite; the best-matching one is offered as "step 0"), keep the best N by `0.4/(1+depth) + 0.3*closeness + 0.3*influence`, then **topologically order** with ties broken by tier then year. Each step says *why* it is there, naming the edge that put it in the path. Experts skip generic foundations unless they are the backbone being adapted. The path is acyclic and ordered by a tested invariant.
5. **Baselines + evaluation setup.** Method papers sharing your problems (ranked by problem specificity and influence) and the benchmarks/backbones those papers use.
6. **Gaps.** Empty mechanism x problem cells next to your idea; mechanism pairs that are each used by >=2 papers, share a specific problem, yet never co-occur; problems with <=2 method papers. Generic problems are excluded or this list is noise (I saw that in the first run).
7. **Momentum.** Per mechanism, its share of proposals from 2023+ against the corpus average: rising / steady / settled, or "too few to call".
8. **Verify externally.** Search queries generated from the entities, plus caveats (corpus limits, unverified edges, low-confidence mapping).

**Worked example** (from the test-suite): *"I want to fine-tune a 4-bit quantized LLaMA with LoRA where the rank is allocated adaptively per layer, to cut GPU memory. Unlike QLoRA, the rank budget is not fixed."* -> mechanisms `adaptive_rank, low_rank_update, quantization`; verdict **NOVEL_COMBINATION**: DyLoRA and AdaLoRA cover {adaptive rank, LoRA}, QLoRA covers {LoRA, quantization}, nothing covers all three; closest = DyLoRA, AdaLoRA, QLoRA, QA-LoRA; untested pairing `adaptive_rank x training_memory`; reading path (expert, 9 steps) runs Intrinsic dimension -> LoRA -> 8-bit optimizers -> LLM.int8 -> GPTQ -> k-bit scaling laws -> DyLoRA -> QLoRA -> AdaLoRA (the three closest papers last). Restating LoRA's own abstract yields `ALREADY_COVERED` with LoRA identified as the originator; a newcomer who names only problems gets `NOT_ASSESSED` but still a foundations-first path.

## 6. Validation, and what is *not* validated

* 23 tests (`python -m unittest tests.test_pipeline`): gold extraction checks; graph invariants (every edge endpoint exists and is a schema relation, all citations go back in time, **every raw citation is typed exactly once**, prerequisite graph acyclic); typed-edge spot checks (`AdaLoRA BUILDS_ON LoRA`, `LLaMA-Adapter EXTENDS Prefix-tuning`, `LoRA COMPARES_WITH_PAPER Adapters`, `LoRA MOTIVATED_BY Intrinsic dimension`); reasoning scenarios for four verdicts; topological order of paths in both levels; output contract; mocked fetcher parsing.
* **Not done:** no held-out extraction evaluation; no comparison to real Semantic Scholar data; no user study of whether the reading paths help; the verdict rules are logical, not calibrated.

## 7. What I chose not to build

* An LLM in the loop (summaries, "explain the idea"). It would hide the structure the brief asks me to own, and every claim here is traceable to an edge.
* Embedding similarity as a second opinion; it could catch paraphrases the lexicon misses, but would mix an unexplainable signal into explainable verdicts.
* Citation-count / PageRank features: not available offline; `influence` over typed prerequisite edges is a more meaningful substitute for *this* purpose.
* A UI. The CLI prints a markdown report or `--json`.
* Ingesting more papers. The point was depth.

## 8. Known limitations

1. Data provenance (top of this file).
2. Lexicon recall: an idea phrased without any lexicon term maps to nothing; the report shows what was recognised so this is visible, and "low confidence" is flagged.
3. Role extraction is sentence-level; title-named base mechanisms are over-credited as proposals (the 3 FPs).
4. `EXTENDS` is coarse: any shared proposed mechanism qualifies, so "QLoRA EXTENDS LLM.int8" (both involve quantization) is a lineage statement, not an engineering one. The edge `evidence` says exactly why it was drawn.
5. Surveys' `COVERS` coverage depends on abstract wording.
6. Lead authors only, so "recurring authors" is weak.
7. The corpus ends mid-2024.

## 9. What I would build next, and why

1. **Run the fetcher, then measure.** Replace the bootstrap with real abstracts/references and hand-label a *held-out* gold set from real abstracts. Everything else is lower value until the extractor has a real error estimate.
2. **Use full text intro/related-work citation contexts** to type edges. The sentence around a citation ("we build on [3]" vs "unlike [3]") is a far better signal than comparing two signatures; the current rules would become the fallback.
3. **Learn the lexicon from the graph (human-in-the-loop).** Surface unmapped frequent n-grams in new papers as candidate mechanisms for me to accept into the ontology; the schema stays mine, maintenance gets cheaper.
4. **Extend to 2025-26 and add temporal queries** ("how did the adaptive-rank neighbourhood change in the last year").
5. **Evaluate the verdicts against reality**: take papers published *after* the corpus date and check whether the system calls their idea novel/combination/covered. That is the most honest test of the whole design.
6. **Weighted reading paths** with a time budget and per-paper reading-effort estimates.

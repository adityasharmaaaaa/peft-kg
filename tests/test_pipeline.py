import json, os, sys, tempfile, unittest
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pefkg.build import build
from pefkg.extract import analyze_paper, analyze_input
from pefkg.reason import Reasoner
from pefkg import ontology as O
import fetch_papers

TMP = tempfile.mkdtemp()
KG = build(os.path.join(ROOT, "data", "papers_raw.json"), TMP, verbose=False)
R = Reasoner(os.path.join(TMP, "kg.json"))
P = KG["nodes"]["papers"]

def roles(pid, role):
    return {m for m, v in P[pid]["signature"]["mechanisms"].items() if v["role"] == role}

# Gold labels: what each paper PROPOSES / merely USES, written from knowledge of the papers themselves.
GOLD_PROPOSES = {
    "lora": {"low_rank_update"}, "adapters": {"adapter_modules"}, "bitfit": {"bias_only"},
    "prefix_tuning": {"soft_prompt"}, "t_few": {"activation_rescaling"}, "adalora": {"adaptive_rank"},
    "unipelt": {"unified_framework"}, "adapterfusion": {"module_composition"}, "galore": {"low_rank_gradient", "optimizer_state_compression"},
    "s_lora": {"serving_system"}, "qlora": {"quantization", "optimizer_state_compression"}, "gptq": {"quantization"},
    "diff_pruning": {"sparse_diff"}, "ewc": {"weight_regularization"}, "grad_checkpoint": {"activation_recomputation"},
    "lora_plus": {"lr_asymmetry"}, "longlora": {"sparse_attention"}, "dora": {"weight_decomposition"},
}
GOLD_USES = {"qlora": {"low_rank_update"}, "adalora": {"low_rank_update"}, "unipelt": {"adapter_modules", "soft_prompt", "low_rank_update"},
             "adapterfusion": {"adapter_modules"}, "s_lora": {"low_rank_update"}}
GOLD_CONTRIB = {"llama": "FOUNDATION_MODEL", "bert": "FOUNDATION_MODEL", "transformer": "FOUNDATION_MODEL", "glue": "BENCHMARK",
                "mmlu": "BENCHMARK", "scaling_laws": "ANALYSIS", "intrinsic_dim": "ANALYSIS", "lora_less_forgets": "ANALYSIS",
                "delta_tuning": "SURVEY", "scaling_down": "SURVEY", "lora": "METHOD", "qlora": "METHOD"}


class TestExtraction(unittest.TestCase):
    def test_proposes_gold(self):
        for pid, gold in GOLD_PROPOSES.items():
            got = roles(pid, "PROPOSES")
            self.assertTrue(gold <= got, f"{pid}: expected {gold} within PROPOSES, got {got}")
    def test_proposes_precision(self):
        tp = fp = 0
        for pid, gold in GOLD_PROPOSES.items():
            got = roles(pid, "PROPOSES"); tp += len(got & gold); fp += len(got - gold)
        self.assertGreaterEqual(tp / (tp + fp), 0.80, f"precision {tp}/{tp+fp}")
    def test_uses_gold(self):
        for pid, gold in GOLD_USES.items():
            self.assertTrue(gold <= roles(pid, "USES"), f"{pid}: {gold} vs {roles(pid, 'USES')}")
    def test_contribution_gold(self):
        for pid, c in GOLD_CONTRIB.items():
            self.assertEqual(P[pid]["contribution"], c, pid)
    def test_name_variants_do_not_become_proposals(self):
        # QLoRA / AdaLoRA / S-LoRA are *named variants*: they build on LoRA, they do not introduce it.
        for pid in ("qlora", "adalora", "s_lora"):
            self.assertNotIn("low_rank_update", roles(pid, "PROPOSES"), pid)
            self.assertIn("low_rank_update", roles(pid, "USES"), pid)
        # 'LoRA adapters' and 'LLaMA-Adapter' are not bottleneck adapter modules
        for pid in ("llama_adapter", "qlora", "s_lora"):
            self.assertNotIn("adapter_modules", roles(pid, "PROPOSES") | roles(pid, "USES"), pid)
    def test_survey_covers_not_proposes(self):
        self.assertFalse(roles("delta_tuning", "PROPOSES")); self.assertTrue(roles("delta_tuning", "COVERS"))
    def test_input_contrast_is_scoped(self):
        a = analyze_input("I use rank allocation per layer. Unlike QLoRA, the rank budget is not fixed.")
        self.assertEqual(a["mechanisms"]["adaptive_rank"]["role"], "IN_IDEA")
        self.assertEqual(a["mechanisms"]["low_rank_update"]["role"], "BASELINE_OR_FOIL")


class TestGraphInvariants(unittest.TestCase):
    def test_edge_endpoints_exist(self):
        nodes = set()
        for kind, d in KG["nodes"].items():
            ent = {"papers": "paper", "mechanisms": "mechanism", "problems": "problem", "concepts": "concept", "models": "model",
                   "benchmarks": "benchmark", "authors": "author", "families": "family"}[kind]
            nodes |= {f"{ent}:{k}" for k in d}
        for e in KG["edges"]:
            self.assertIn(e["source"], nodes); self.assertIn(e["target"], nodes)
            self.assertIn(e["relation"], O.RELATION_TYPES)
    def test_citations_go_back_in_time(self):
        for e in KG["edges"]:
            if e["source"].startswith("paper:") and e["target"].startswith("paper:"):
                self.assertLessEqual(P[e["target"][6:]]["year"], P[e["source"][6:]]["year"])
    def test_every_raw_citation_is_typed_exactly_once(self):
        with open(os.path.join(ROOT, "data", "papers_raw.json")) as f: raw = json.load(f)["papers"]
        n = sum(len(p["references"]) for p in raw)
        typed = [e for e in KG["edges"] if e["source"].startswith("paper:") and e["target"].startswith("paper:")]
        self.assertEqual(len(typed), n)
    def test_prerequisite_graph_is_acyclic(self):
        adj = defaultdict(set)
        for a, rel, b, _ in R.prereq_edges: adj[a].add(b)
        state = {}
        def dfs(x):
            state[x] = 1
            for y in adj[x]:
                self.assertNotEqual(state.get(y), 1, f"cycle through {x}->{y}")
                if y not in state: dfs(y)
            state[x] = 2
        for x in list(adj):
            if x not in state: dfs(x)
    def test_typed_edge_examples(self):
        def rel(a, b): return {e["relation"] for e in KG["edges"] if e["source"] == f"paper:{a}" and e["target"] == f"paper:{b}"}
        self.assertEqual(rel("adalora", "lora"), {"BUILDS_ON"})
        self.assertEqual(rel("llama_adapter", "prefix_tuning"), {"EXTENDS"})
        self.assertEqual(rel("lora", "adapters"), {"COMPARES_WITH_PAPER"})
        self.assertEqual(rel("lora", "intrinsic_dim"), {"MOTIVATED_BY"})
        self.assertEqual(rel("lora", "bert"), {"USES_BACKBONE"})
        self.assertEqual(rel("delta_tuning", "lora"), {"SURVEYS"})
    def test_originators(self):
        self.assertEqual(KG["nodes"]["mechanisms"]["low_rank_update"]["first_in_corpus"], "lora")
        self.assertEqual(KG["nodes"]["mechanisms"]["adapter_modules"]["first_in_corpus"], "adapters")


class TestReasoning(unittest.TestCase):
    def test_already_covered(self):
        out = R.run("We freeze pretrained weights and inject trainable low-rank decomposition matrices, so GPT-3 fine-tuning needs less GPU memory and adds no inference latency.")
        self.assertEqual(out["novelty_assessment"]["verdict"], "ALREADY_COVERED")
        self.assertEqual(out["closest_prior_work"][0]["paper"], "lora")
    def test_novel_combination(self):
        out = R.run("4-bit quantized LLaMA with LoRA where the rank is allocated adaptively per layer, to cut GPU memory. Unlike QLoRA the rank budget is not fixed.")
        self.assertEqual(out["novelty_assessment"]["verdict"], "NOVEL_COMBINATION")
        self.assertTrue({"adalora", "qlora"} <= {c["paper"] for c in out["closest_prior_work"]})
    def test_novel_application(self):
        out = R.run("Use prefix tuning with soft prompts to handle long-context adaptation of sequence length.")
        self.assertEqual(out["novelty_assessment"]["verdict"], "NOVEL_APPLICATION")
    def test_newcomer_not_assessed_but_gets_a_path(self):
        out = R.run("I'm new and want cheaper fine-tuning memory without forgetting.")
        self.assertEqual(out["novelty_assessment"]["verdict"], "NOT_ASSESSED")
        self.assertGreater(len(out["reading_path"]["steps"]), 3)
    def test_reading_path_is_topologically_ordered(self):
        for level in ("beginner", "expert"):
            out = R.run("LoRA with quantization to reduce memory", level=level)
            pos = {s["paper"]: s["step"] for s in out["reading_path"]["steps"]}
            for s in out["reading_path"]["steps"]:
                for dep in s["read_after"]:
                    self.assertLess(pos[dep], s["step"], f"{dep} must precede {s['paper']}")
    def test_expert_path_is_shorter_than_beginner(self):
        txt = "LoRA with quantization to reduce memory on LLaMA"
        b = len(R.run(txt, level="beginner")["reading_path"]["steps"]); e = len(R.run(txt, level="expert")["reading_path"]["steps"])
        self.assertLessEqual(e, b)
        foundations = [s for s in R.run(txt, level="expert")["reading_path"]["steps"] if s["tier"] == "FOUNDATION" and s["paper"] != "llama"]
        self.assertEqual(foundations, [])
    def test_output_contract(self):
        out = R.run("adapters for multi-task transfer")
        for k in ("understanding", "novelty_assessment", "closest_prior_work", "reading_path", "baselines_and_evaluation",
                  "open_gaps", "field_momentum", "queries_to_verify_externally", "caveats"):
            self.assertIn(k, out)
        json.dumps(out)   # serialisable


class TestFetchParsing(unittest.TestCase):
    def test_arxiv_year(self):
        self.assertEqual(fetch_papers.arxiv_year({"ArXiv": "1810.04805"}), 2018)
        self.assertIsNone(fetch_papers.arxiv_year({}))
    def test_parse_paper_and_refs(self):
        rec = {"paperId": "abc", "title": "LoRA: Low-Rank Adaptation of Large Language Models", "year": 2022, "abstract": "An abstract.",
               "authors": [{"name": "Edward J. Hu"}, {"name": "Y. Shen"}], "externalIds": {"ArXiv": "2106.09685"}}
        p = fetch_papers.parse_paper(rec, "lora", "LoRA", "x")
        self.assertEqual((p["year"], p["authors"], p["abstract_source"]), (2021, ["Edward J. Hu"], "semantic_scholar"))
        self.assertEqual(fetch_papers.refs_from({"data": [{"citedPaper": {"paperId": "p1"}}, {"citedPaper": {"paperId": None}}], "next": None}), (["p1"], None))
    def test_tldr_fallback(self):
        p = fetch_papers.parse_paper({"paperId": "a", "title": "t", "year": 2020, "abstract": None, "tldr": {"text": "short"}, "authors": []}, "x", "x", "t")
        self.assertEqual((p["abstract"], p["abstract_source"]), ("short", "semantic_scholar_tldr"))


if __name__ == "__main__":
    unittest.main(verbosity=1)

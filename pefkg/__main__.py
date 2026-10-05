"""
CLI.
  python -m pefkg build [--raw data/papers_raw.json] [--out knowledge_state]
  python -m pefkg ask "your idea or situation" [--level beginner|expert] [--json] [--file idea.txt]
  python -m pefkg inspect <paper_id | mechanism_id | problem_id>
"""
import argparse, json, os, sys
from .build import build
from .reason import Reasoner
from .render import render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pefkg")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--raw", default=os.path.join(ROOT, "data", "papers_raw.json")); b.add_argument("--out", default=os.path.join(ROOT, "knowledge_state"))
    a = sub.add_parser("ask"); a.add_argument("text", nargs="?"); a.add_argument("--file"); a.add_argument("--level", choices=["beginner", "expert"], default="beginner")
    a.add_argument("--json", action="store_true"); a.add_argument("--kg", default=os.path.join(ROOT, "knowledge_state", "kg.json")); a.add_argument("--max-papers", type=int, default=12)
    i = sub.add_parser("inspect"); i.add_argument("entity"); i.add_argument("--kg", default=os.path.join(ROOT, "knowledge_state", "kg.json"))
    args = ap.parse_args(argv)

    if args.cmd == "build":
        build(args.raw, args.out)
    elif args.cmd == "ask":
        text = open(args.file).read() if args.file else (args.text or sys.stdin.read())
        out = Reasoner(args.kg).run(text, level=args.level, max_papers=args.max_papers)
        print(json.dumps(out, indent=2) if args.json else render(out))
    elif args.cmd == "inspect":
        kg = json.load(open(args.kg)); N = kg["nodes"]
        for kind in ("papers", "mechanisms", "problems", "concepts", "models", "benchmarks"):
            if args.entity in N[kind]:
                print(json.dumps(N[kind][args.entity], indent=2))
                ent = {"papers": "paper", "mechanisms": "mechanism", "problems": "problem", "concepts": "concept", "models": "model", "benchmarks": "benchmark"}[kind]
                key = f"{ent}:{args.entity}"
                print("\n-- edges --")
                for e in kg["edges"]:
                    if key in (e["source"], e["target"]):
                        print(f"{e['source']} -[{e['relation']}]-> {e['target']}   ({e['evidence']})")
                return
        print("not found"); sys.exit(1)


if __name__ == "__main__":
    main()

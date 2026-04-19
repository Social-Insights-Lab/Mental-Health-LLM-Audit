import json, os, argparse
from pathlib import Path
from collections import Counter, defaultdict

MERGED_DIR = "../results_new/merged"
DISORDERS_FILE = "../filtered_combined_disorders.json"
EXPECTED_PER_GROUP = 45


def iter_merged(merged_dir):
    for fname in sorted(os.listdir(merged_dir)):
        if fname.startswith("merged_") and fname.endswith(".jsonl"):
            model = fname.replace("merged_", "").replace("_generations.jsonl", "")
            yield model, os.path.join(merged_dir, fname)


def cmd_structure(args):
    print(f"Verifying structure (expected: 132 groups × {EXPECTED_PER_GROUP} entries)\n")
    all_ok = True
    for model, fpath in iter_merged(args.merged_dir):
        counts = Counter()
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        counts[json.loads(line).get("group", "").strip()] += 1
                    except json.JSONDecodeError:
                        continue
        issues = {g: c for g, c in counts.items() if c != EXPECTED_PER_GROUP}
        status = "OK" if not issues else "FAIL"
        print(f"[{status}] {model}: {len(counts)} groups, {sum(counts.values())} entries")
        if issues:
            all_ok = False
            for g, c in issues.items():
                print(f"       {g}: {c}")
    print("\nAll files correct." if all_ok else "\nSome files have issues.")


def cmd_nulls(args):
    print("Analyzing null evaluations in merged files\n")
    for model, fpath in iter_merged(args.merged_dir):
        total, any_null, all_null = 0, 0, 0
        null_counts = defaultdict(int)
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                evals = json.loads(line).get("evaluations", {})
                nulls = [e for e in ["Claude", "OpenAI", "Deepseek"] if evals.get(e) is None]
                for e in nulls:
                    null_counts[e] += 1
                if nulls:
                    any_null += 1
                if len(nulls) == 3:
                    all_null += 1
        print(f"{model}: total={total}, any_null={any_null}, all_null={all_null} "
              f"| Claude={null_counts['Claude']} OpenAI={null_counts['OpenAI']} Deepseek={null_counts['Deepseek']}")


def cmd_disorders(args):
    expected = set(d.strip() for d in json.loads(Path(args.disorders_file).read_text(encoding="utf-8")))
    first_file = next((p for _, p in iter_merged(args.merged_dir)), None)
    if not first_file:
        print("No merged files found.")
        return
    actual = set()
    with open(first_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                actual.add(json.loads(line).get("group", "").strip())
    missing = expected - actual
    extra = actual - expected
    print(f"Expected: {len(expected)}  |  Found: {len(actual)}  |  Missing: {len(missing)}  |  Extra: {len(extra)}")
    if missing:
        print("\nMissing:", sorted(missing))
    if extra:
        print("\nExtra:", sorted(extra))
    if not missing and not extra:
        print("Perfect match.")


def cmd_missing(args):
    all_groups = {}
    for model, fpath in iter_merged(args.merged_dir):
        groups = set()
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    groups.add(json.loads(line).get("group", "").strip())
        all_groups[model] = groups
        print(f"{model}: {len(groups)} groups")

    universe = set().union(*all_groups.values())
    print(f"\nUnion across all files: {len(universe)} groups")
    for model, groups in all_groups.items():
        missing = universe - groups
        if missing:
            print(f"\n{model} missing: {sorted(missing)}")


def main():
    parser = argparse.ArgumentParser(description="Validate merged data files.")
    parser.add_argument("--merged_dir", default=MERGED_DIR)
    parser.add_argument("--disorders_file", default=DISORDERS_FILE)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("structure", help="Verify 45 entries per group per file")
    sub.add_parser("nulls", help="Count null evaluations per model")
    sub.add_parser("disorders", help="Compare expected vs actual disorder groups")
    sub.add_parser("missing", help="Find groups missing from any file")
    args = parser.parse_args()
    {"structure": cmd_structure, "nulls": cmd_nulls,
     "disorders": cmd_disorders, "missing": cmd_missing}[args.cmd](args)


if __name__ == "__main__":
    main()

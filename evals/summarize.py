import argparse
from collections import defaultdict

from langsmith import Client

import app.config  # noqa: F401 — import triggers .env autoload before Client() reads LANGSMITH_API_KEY

# (feedback key, column header, width)
COLUMNS = (
    ("correctness", "corr", 5),
    ("citation_support", "cite", 5),
    ("refs_resolved", "refs", 5),
    ("findings_grounded", "grnd", 5),
    ("answer_refs_valid", "aref", 5),
    ("latency_s", "lat_s", 7),
    ("total_tokens", "tok", 7),
    ("search_calls", "srch", 5),
    ("num_sources", "src", 4),
)
KEYS = tuple(key for key, _, _ in COLUMNS)
DEFAULT_ARMS = ["skill-on-iters3", "skill-off-iters3", "skill-on-iters1", "skill-off-iters1"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Average LangSmith feedback per experiment prefix (latest project each)."
    )
    parser.add_argument(
        "prefixes",
        nargs="*",
        default=DEFAULT_ARMS,
        help="experiment prefixes, as passed to run_evals --experiment-prefix",
    )
    args = parser.parse_args(argv)
    width = max(22, *(len(p) for p in args.prefixes))
    client = Client()
    print(f"{'arm':{width}s} {'n':>2} " + " ".join(f"{h:>{w}}" for _, h, w in COLUMNS))
    projects = list(client.list_projects())
    for arm in args.prefixes:
        matches = [p for p in projects if p.name.startswith(arm)]
        if not matches:
            print(f"{arm:{width}s}  MISSING")
            continue
        project = sorted(matches, key=lambda p: p.start_time)[-1]
        stats: dict[str, list[float]] = defaultdict(list)
        n = 0
        for run in client.list_runs(project_id=project.id, is_root=True):
            n += 1
            for fb in client.list_feedback(run_ids=[run.id]):
                if fb.key in KEYS and fb.score is not None:
                    stats[fb.key].append(float(fb.score))
        row = {k: (round(sum(v) / len(v), 2) if v else None) for k, v in stats.items()}
        print(
            f"{arm:{width}s} {n:>2} "
            + " ".join(f"{str(row.get(k)):>{w}}" for k, _, w in COLUMNS)
        )


if __name__ == "__main__":
    main()

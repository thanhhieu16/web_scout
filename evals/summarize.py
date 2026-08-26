from collections import defaultdict

from langsmith import Client

KEYS = (
    "correctness", "citation_support", "latency_s",
    "total_tokens", "search_calls", "num_sources",
)
ARMS = ["skill-on-iters3", "skill-off-iters3", "skill-on-iters1", "skill-off-iters1"]

client = Client()
print(
    f"{'arm':22s} {'n':>2} {'corr':>5} {'cite':>5} "
    f"{'lat_s':>7} {'tok':>7} {'srch':>5} {'src':>4}"
)
for arm in ARMS:
    matches = [p for p in client.list_projects() if p.name.startswith(arm)]
    if not matches:
        print(f"{arm:22s}  MISSING")
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
        f"{arm:22s} {n:>2} "
        f"{str(row.get('correctness')):>5} {str(row.get('citation_support')):>5} "
        f"{str(row.get('latency_s')):>7} {str(row.get('total_tokens')):>7} "
        f"{str(row.get('search_calls')):>5} {str(row.get('num_sources')):>4}"
    )

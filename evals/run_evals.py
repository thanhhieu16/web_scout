import argparse
import itertools
import json
from functools import lru_cache
from pathlib import Path

from langsmith import Client

DATASET_PATH = Path(__file__).parent / "dataset.json"
DATASET_NAME = "webscout-evals-v1"


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _graph():
    from app.graph import build_graph

    return build_graph()


def target(inputs: dict) -> dict:
    s = _graph().invoke(
        {
            "question": inputs["question"],
            "iteration": 0,
            "max_iterations": 3,
        }
    )
    return {
        "answer": s.get("answer", ""),
        "sources": s.get("sources", []),
        "search_calls": s.get("search_calls", 0),
    }


def ensure_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        return client.read_dataset(dataset_name=DATASET_NAME)
    ds = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": r["question"]} for r in load_dataset()],
        outputs=[{"reference_notes": r["reference_notes"]} for r in load_dataset()],
    )
    return ds


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-prefix", default="webscout-run")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    from evals.evaluators import (
        citation_support_evaluator,
        correctness_evaluator,
        metrics_evaluator,
    )

    client = Client()
    ensure_dataset(client)
    if args.limit:
        data = list(
            itertools.islice(client.list_examples(dataset_name=DATASET_NAME), args.limit)
        )
    else:
        data = DATASET_NAME
    client.evaluate(
        target,
        data=data,
        evaluators=[
            correctness_evaluator,
            citation_support_evaluator,
            metrics_evaluator,
        ],
        experiment_prefix=args.experiment_prefix,
        max_concurrency=2,
    )


if __name__ == "__main__":
    main()

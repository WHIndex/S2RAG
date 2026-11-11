import argparse
import json
from pathlib import Path

from .S2RAG import S2RAG
from .profile import build_s2rag_config


def _load_input(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents") or []
    questions = payload.get("questions") or []
    if not isinstance(documents, list) or not all(isinstance(row, str) for row in documents):
        raise ValueError("documents must be a list of strings")
    if not isinstance(questions, list) or not all(isinstance(row, str) for row in questions):
        raise ValueError("questions must be a list of strings")
    if not documents or not questions:
        raise ValueError("documents and questions must be non-empty")
    return documents, questions


def main() -> int:
    parser = argparse.ArgumentParser(prog="s2rag")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("answers.json"))
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--retrieval-backend",
        choices=("graph_ppr", "nv_embed_dense"),
        default="graph_ppr",
    )
    parser.add_argument("--calibration-path", default="")
    args = parser.parse_args()

    documents, questions = _load_input(args.input)
    config = build_s2rag_config(
        save_dir=args.save_dir,
        llm_name=args.model,
        llm_base_url=args.base_url,
        retrieval_backend=args.retrieval_backend,
        calibration_path=args.calibration_path,
    )
    engine = S2RAG(global_config=config)
    engine.index(documents)
    solutions, _, metadata = engine.rag_qa(questions)
    rows = [
        {
            "question": solution.question,
            "answer": solution.answer,
            "metadata": item,
        }
        for solution, item in zip(solutions, metadata)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

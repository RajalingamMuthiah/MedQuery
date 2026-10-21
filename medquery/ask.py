"""Answer one question from the terminal: uv run python -m medquery.ask "question"."""

import argparse
import sys

from medquery import config
from medquery.generation.answer import NOT_MEDICAL_ADVICE, Answer, answer
from medquery.llm import LLMError, get_client
from medquery.retrieval.dense import Retriever, load_chunks
from medquery.retrieval.embed import FastEmbedder


def format_answer(result: Answer, show_context: bool = False) -> str:
    lines = [result.text, ""]
    by_id = {h.chunk["id"]: h.chunk for h in result.hits}
    if result.cited_ids:
        lines.append("Sources:")
        for chunk_id in result.cited_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                lines.append(f"  [{chunk_id}] (not in the retrieved passages)")
                continue
            pages = f"p.{chunk['page_start']}" + (
                f"-{chunk['page_end']}" if chunk["page_end"] != chunk["page_start"] else ""
            )
            note = f", as of {chunk['as_of']} (SUPERSEDED)" if chunk.get("superseded") else ""
            section = f", {chunk['section']}" if chunk.get("section") else ""
            lines.append(f"  [{chunk_id}] {chunk['title']}, {pages}{section}{note}")
        lines.append("")
    if show_context:
        lines.append("Retrieved passages:")
        lines += [f"  {h.rank}. [{h.chunk['id']}] score={h.score:.3f}" for h in result.hits]
        lines.append("")
    usage = result.completion.usage
    lines.append(
        f"({result.completion.model}; retrieval {result.retrieval_s * 1000:.0f} ms, generation "
        f"{result.completion.latency_s:.2f} s{', cached' if result.completion.cached else ''}; "
        f"{usage.get('prompt_tokens', 0)} in / {usage.get('completion_tokens', 0)} out tokens)"
    )
    lines.append(NOT_MEDICAL_ADVICE)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m medquery.ask")
    parser.add_argument("question", nargs="+")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args(argv)
    question = " ".join(args.question)
    try:
        retriever = Retriever(FastEmbedder(), load_chunks())
        result = answer(question, retriever, get_client(), k=args.k)
    except (LLMError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_answer(result, args.show_context))
    return 0


if __name__ == "__main__":
    sys.exit(main())

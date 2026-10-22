"""Judge validation: a stratified sample of judge decisions for hand labelling, and the
agreement (% and Cohen's kappa) between those labels and the judge.

The judge's verdicts are kept out of the review CSV (in a separate key file) so the labeller
is not anchored by them.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from medquery.eval.golden import GOLDEN_DIR
from medquery.eval.stats import agreement, bootstrap_ci, cohen_kappa
from medquery.fileio import atomic_write

REVIEW_VERSION = "v1"
CLAIM_LABELS = ("supported", "unsupported")
ANSWER_LABELS = ("correct", "partial", "incorrect")
N_CLAIMS, N_ANSWERS = 40, 20
MAX_CLAIMS_PER_ITEM = 3
SEED = 20261022
COLUMNS = [
    "row_id",
    "unit",
    "topic",
    "question",
    "passages",
    "answer",
    "reference_answer",
    "claim",
    "human_label",
    "notes",
]


def review_paths(version: str = REVIEW_VERSION, root: Path = GOLDEN_DIR) -> dict[str, Path]:
    stem = f"judge_review_{version}"
    return {
        "csv": root / f"{stem}.csv",
        "key": root / f"{stem}.key.json",
        "labels": root / f"{stem}.labels.jsonl",
        "meta": root / f"{stem}.labels.meta.json",
    }


def _pick(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n units, spread round-robin across topics (seeded)."""
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for unit in pool:
        by_topic[unit["topic"]].append(unit)
    topics = sorted(by_topic)
    rng.shuffle(topics)
    for t in topics:
        rng.shuffle(by_topic[t])
    chosen: list[dict] = []
    while len(chosen) < n and any(by_topic[t] for t in topics):
        for t in topics:
            if by_topic[t] and len(chosen) < n:
                chosen.append(by_topic[t].pop())
    return chosen


def _allocate(pools: dict[str, list[dict]], n: int, minority: tuple[str, ...], rng) -> list[dict]:
    """Minority labels get at least a third of the sample where available; the rest fills from
    the majority label, then from any label with units left."""
    available = sum(len(pools[m]) for m in minority)
    want_minority = min(available, math.ceil(n / 3))
    chosen: list[dict] = []
    for position, m in enumerate(minority):  # quota shared in proportion to availability
        if position == len(minority) - 1:
            share = want_minority - len(chosen)
        else:
            share = round(want_minority * len(pools[m]) / available) if available else 0
        chosen += _pick(pools[m], min(share, len(pools[m])), rng)
    majority = [u for label, pool in pools.items() if label not in minority for u in pool]
    chosen += _pick(majority, n - len(chosen), rng)
    taken = {id(u) for u in chosen}
    rest = [u for pool in pools.values() for u in pool if id(u) not in taken]
    chosen += _pick(rest, n - len(chosen), rng)  # only if the majority labels ran out
    return chosen


MAX_PROBES = 16


def probe_units(
    records: list[dict],
    llm,
    chunks: dict[str, dict],
    need_unsupported: int,
    seed: int = SEED,
    max_probes: int = MAX_PROBES,
) -> tuple[list[dict], list]:
    """Mismatched-context probes: re-judge a dev answer against the passages retrieved for a
    dev question from a different topic, so its claims are rarely supported by what is shown.
    Natural runs hallucinate too rarely to validate the judge on unsupported claims; probes
    supply them. Units are labelled against the passages shown, like natural units."""
    from medquery.eval.judge import JudgeFailed, judge
    from medquery.retrieval.dense import Hit

    rng = random.Random(seed + 1)
    answered = [r for r in sorted(records, key=lambda r: r["id"]) if r["judge"] and r["answer"]]
    rng.shuffle(answered)
    units: list[dict] = []
    completions: list = []
    for r in answered[:max_probes]:
        if sum(u["judge_label"] == "unsupported" for u in units) >= need_unsupported:
            break
        donors = [d for d in answered if d["topic"] != r["topic"]]
        donor = donors[rng.randrange(len(donors))]
        hits = [Hit(chunks[cid], 0.0, n) for n, cid in enumerate(donor["context_ids"], 1)]
        try:
            verdict, calls = judge(r["question"], hits, r["answer"], r["reference_answer"], llm)
        except JudgeFailed as exc:
            completions += exc.completions
            continue
        completions += calls
        for index, claim in enumerate(verdict.claims):
            label = "supported" if claim["supported_by_context"] else "unsupported"
            units.append(
                {
                    "unit": "claim",
                    "record": r,
                    "claim_index": index,
                    "claim": claim["text"],
                    "topic": r["topic"],
                    "judge_label": label,
                    "context_ids": donor["context_ids"],
                    "probe": True,
                    "donor_id": donor["id"],
                }
            )
    return units, completions


def sample(
    records: list[dict],
    n_claims: int = N_CLAIMS,
    n_answers: int = N_ANSWERS,
    seed: int = SEED,
    probes: list[dict] | None = None,
) -> list[dict]:
    """Units from judged answers: claims stratified by topic x support verdict, answers by
    topic x correctness verdict. Each unit keeps the judge's label (for the hidden key)."""
    rng = random.Random(seed)
    claims: dict[str, list[dict]] = {label: [] for label in CLAIM_LABELS}
    answers: dict[str, list[dict]] = {label: [] for label in ANSWER_LABELS}
    for r in sorted(records, key=lambda r: r["id"]):
        if r["abstained"] or not r["judge"]:
            continue
        indexed = list(enumerate(r["judge"]["claims"]))
        rng.shuffle(indexed)
        for index, claim in indexed[:MAX_CLAIMS_PER_ITEM]:
            label = "supported" if claim["supported_by_context"] else "unsupported"
            claims[label].append(
                {
                    "unit": "claim",
                    "record": r,
                    "claim_index": index,
                    "claim": claim["text"],
                    "topic": r["topic"],
                    "judge_label": label,
                    "context_ids": r["context_ids"],
                    "probe": False,
                }
            )
        if r["answerable"]:
            label = r["judge"]["correctness"]
            answers[label].append(
                {
                    "unit": "answer",
                    "record": r,
                    "claim_index": None,
                    "claim": "",
                    "topic": r["topic"],
                    "judge_label": label,
                    "context_ids": r["context_ids"],
                    "probe": False,
                }
            )
    # Probes (mismatched context) only add to the unsupported pool; supported claims come from
    # natural answers only.
    claims["unsupported"] += [p for p in probes or [] if p["judge_label"] == "unsupported"]
    units = _allocate(claims, n_claims, ("unsupported",), rng)
    units += _allocate(answers, n_answers, ("partial", "incorrect"), rng)
    rng.shuffle(units)  # mix units and verdicts so labels are not grouped
    return units


def review_rows(units: list[dict], chunk_text: dict[str, str]) -> tuple[list[dict], dict]:
    rows, key = [], {}
    for n, unit in enumerate(units, 1):
        row_id = f"r{n:03d}"
        r = unit["record"]
        passage_ids = unit.get("context_ids", r["context_ids"])
        passages = "\n---\n".join(f"[{cid}]\n{chunk_text.get(cid, '')}" for cid in passage_ids)
        rows.append(
            {
                "row_id": row_id,
                "unit": unit["unit"],
                "topic": r["topic"],
                "question": r["question"],
                "passages": passages,
                "answer": r["answer"],
                "reference_answer": r["reference_answer"] if unit["unit"] == "answer" else "",
                "claim": unit["claim"],
                "human_label": "",
                "notes": "",
            }
        )
        key[row_id] = {
            "unit": unit["unit"],
            "item_id": r["id"],
            "claim_index": unit["claim_index"],
            "judge_label": unit["judge_label"],
            "probe": unit.get("probe", False),
            "donor_id": unit.get("donor_id"),
        }
    return rows, key


def write_review(rows: list[dict], key: dict, meta: dict, paths: dict[str, Path]) -> None:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    with atomic_write(paths["csv"]) as f:
        f.write("\ufeff" + buffer.getvalue())
    with atomic_write(paths["key"]) as f:
        json.dump({"meta": meta, "rows": key}, f, indent=2, ensure_ascii=False)


def agreement_block(labels: list[dict], judge_prompt: str, judge_model: str, meta: dict) -> dict:
    """Agreement and kappa per unit type, each with a 95% bootstrap CI. Marked STALE when the
    labels were collected for a different judge model or prompt than the one in use."""
    out = {
        "labels_sha256": meta.get("sha256"),
        "stale": meta.get("judge_prompt") != judge_prompt or meta.get("judge_model") != judge_model,
        "labelled_for": f"{meta.get('judge_model')} {meta.get('judge_prompt')}",
    }
    for unit in ("claim", "answer"):
        pairs = [(x["human_label"], x["judge_label"]) for x in labels if x["unit"] == unit]
        human, judged = [p[0] for p in pairs], [p[1] for p in pairs]

        def pct(sample):
            return agreement([p[0] for p in sample], [p[1] for p in sample])

        def kappa(sample):
            return cohen_kappa([p[0] for p in sample], [p[1] for p in sample])

        counts: dict[str, int] = defaultdict(int)
        for h in human:
            counts[h] += 1
        out[unit] = {
            "n": len(pairs),
            "n_probe": sum(1 for x in labels if x["unit"] == unit and x.get("probe")),
            "agreement": bootstrap_ci(pairs, pct),
            "kappa": bootstrap_ci(pairs, kappa),
            "human_class_counts": dict(sorted(counts.items())),
            "judge_class_counts": {lab: judged.count(lab) for lab in sorted(set(judged))},
        }
    return out


def load_agreement(
    judge_prompt: str, judge_model: str, version: str = REVIEW_VERSION, root: Path = GOLDEN_DIR
) -> dict | None:
    """The agreement block from frozen labels, or None if the labels are not frozen yet."""
    paths = review_paths(version, root)
    if not paths["labels"].exists() or not paths["meta"].exists():
        return None
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    labels = [json.loads(line) for line in paths["labels"].read_text(encoding="utf-8").splitlines()]
    return agreement_block(labels, judge_prompt, judge_model, meta)

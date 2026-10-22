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


REVISIONS = ("", "r2", "r3", "r4")  # "" is the first frozen label set; later ones correct it


def review_paths(
    version: str = REVIEW_VERSION, root: Path = GOLDEN_DIR, revision: str = ""
) -> dict[str, Path]:
    stem = f"judge_review_{version}"
    suffix = f"_{revision}" if revision else ""
    return {
        "csv": root / f"{stem}.csv",
        "key": root / f"{stem}.key.json",
        "labels": root / f"{stem}.labels{suffix}.jsonl",
        "meta": root / f"{stem}.labels{suffix}.meta.json",
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
        f.write("\n")


def agreement_block(labels: list[dict], judge_prompt: str, judge_model: str, meta: dict) -> dict:
    """Agreement and kappa per unit type, each with a 95% bootstrap CI. Marked STALE when the
    labels were collected for a different judge model or prompt than the one in use."""
    out = {
        "labels_sha256": meta.get("sha256"),
        "labeller": meta.get("labeller"),
        "stale": meta.get("judge_prompt") != judge_prompt or meta.get("judge_model") != judge_model,
        "labelled_for": f"{meta.get('judge_model')} {meta.get('judge_prompt')}",
    }
    groups = {
        "claim": lambda x: x["unit"] == "claim",
        "claim_natural": lambda x: x["unit"] == "claim" and not x.get("probe"),
        "claim_probe": lambda x: x["unit"] == "claim" and x.get("probe"),
        "answer": lambda x: x["unit"] == "answer",
    }
    out["labels_revision"] = meta.get("revision") or "v1"
    out["supersedes"] = meta.get("supersedes")
    natural_unsupported = [
        x
        for x in labels
        if x["unit"] == "claim" and not x.get("probe") and x["human_label"] == "unsupported"
    ]
    out["natural_unsupported_caught"] = {
        "human_unsupported": len(natural_unsupported),
        "judge_caught": sum(x["judge_label"] == "unsupported" for x in natural_unsupported),
    }
    for unit, member in groups.items():
        # A judge verdict can be missing (None) when a re-judged claim could not be matched.
        pairs = [
            (x["human_label"], x["judge_label"])
            for x in labels
            if member(x) and x["judge_label"] is not None
        ]
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
            "n_probe": sum(1 for x in labels if member(x) and x.get("probe")),
            "agreement": bootstrap_ci(pairs, pct),
            "kappa": bootstrap_ci(pairs, kappa),
            "human_class_counts": dict(sorted(counts.items())),
            "judge_class_counts": {lab: judged.count(lab) for lab in sorted(set(judged))},
        }
    return out


def latest_revision(version: str = REVIEW_VERSION, root: Path = GOLDEN_DIR) -> str | None:
    found = None
    for revision in REVISIONS:
        paths = review_paths(version, root, revision)
        if paths["labels"].exists() and paths["meta"].exists():
            found = revision
    return found


def load_labels(version: str, root: Path, revision: str) -> tuple[list[dict], dict]:
    paths = review_paths(version, root, revision)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    text = paths["labels"].read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()], meta


def measurement_path(version: str, root: Path, judge_version: str, revision: str) -> Path:
    return root / f"judge_review_{version}.measure_{judge_version}_{revision or 'v1'}.json"


def load_agreement(
    judge_prompt: str, judge_model: str, version: str = REVIEW_VERSION, root: Path = GOLDEN_DIR
) -> dict | None:
    """Agreement of the judge in use with the latest frozen labels, or None if none are frozen.

    The labels carry the verdicts of the judge they were sampled with. For another judge
    version, a saved measurement on the same frozen labels supplies its verdicts; without one
    the block is marked STALE."""
    revision = latest_revision(version, root)
    if revision is None:
        return None
    labels, meta = load_labels(version, root, revision)
    measured = None
    if meta["judge_prompt"] != judge_prompt or meta["judge_model"] != judge_model:
        path = measurement_path(version, root, judge_prompt.split(":")[0], revision)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            same_judge = data["judge_prompt"] == judge_prompt and data["judge_model"] == judge_model
            if same_judge and data["labels_sha256"] == meta["sha256"]:
                measured = data
                labels = [{**x, "judge_label": data["row_labels"].get(x["row_id"])} for x in labels]
                meta = {**meta, "judge_prompt": judge_prompt, "judge_model": judge_model}
    block = agreement_block(labels, judge_prompt, judge_model, meta)
    block["measured_on_frozen_labels"] = measured is not None
    block["unmatched"] = measured["unmatched"] if measured else []
    return block


def _ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, " ".join(a.lower().split()), " ".join(b.lower().split())).ratio()


MATCH_RATIO = 0.6  # a re-judged claim counts as the labelled one at this text similarity


def measure_judge(
    judge_version: str,
    judge_model: str,
    labels: list[dict],
    claim_text: dict[str, str],
    records: dict[str, dict],
    chunks: dict[str, dict],
    llm,
) -> tuple[dict[str, str | None], list[str], list]:
    """Re-judge the labelled answers with another judge version, on the same passages as the
    labelled units (donor passages for probes), and read off its verdict for each unit.

    The deployed prompt re-extracts claims, so each labelled claim is matched to the closest
    re-extracted claim (text similarity >= MATCH_RATIO); unmatched units are returned, not
    guessed. Returns (row_id -> verdict or None, unmatched row ids, completions)."""
    from medquery.eval.judge import JudgeFailed, judge
    from medquery.retrieval.dense import Hit

    groups: dict[tuple[str, str | None], list[dict]] = defaultdict(list)
    for x in labels:
        groups[(x["item_id"], x.get("donor_id") if x.get("probe") else None)].append(x)
    verdicts: dict[str, str | None] = {}
    unmatched: list[str] = []
    completions: list = []
    for (item_id, donor_id), units in sorted(
        groups.items(), key=lambda g: (g[0][0], g[0][1] or "")
    ):
        record = records[item_id]
        context = records[donor_id]["context_ids"] if donor_id else record["context_ids"]
        hits = [Hit(chunks[cid], 0.0, n) for n, cid in enumerate(context, 1)]
        try:
            verdict, calls = judge(
                record["question"],
                hits,
                record["answer"],
                record["reference_answer"],
                llm,
                version=judge_version,
                model=judge_model,
            )
        except JudgeFailed as exc:
            completions += exc.completions
            for x in units:
                verdicts[x["row_id"]] = None
                unmatched.append(x["row_id"])
            continue
        completions += calls
        for x in units:
            if x["unit"] == "answer":
                verdicts[x["row_id"]] = verdict.correctness
                continue
            scored = [(_ratio(claim_text[x["row_id"]], c["text"]), c) for c in verdict.claims]
            best = max(scored, key=lambda s: s[0], default=(0.0, None))
            if best[0] >= MATCH_RATIO:
                verdicts[x["row_id"]] = (
                    "supported" if best[1]["supported_by_context"] else "unsupported"
                )
            else:
                verdicts[x["row_id"]] = None
                unmatched.append(x["row_id"])
    return verdicts, sorted(unmatched), completions


ALLOWED = {"claim": CLAIM_LABELS, "answer": ANSWER_LABELS}


class LabelError(Exception):
    pass


def freeze_labels(
    labeller: str, version: str = REVIEW_VERSION, root: Path = GOLDEN_DIR, revision: str = ""
) -> dict:
    """Join hand labels (CSV) with the hidden judge verdicts (key) and freeze them.

    A revision (r2, ...) is a corrected label set frozen next to the earlier ones, which are
    never changed; its meta names the set it supersedes and the rows that changed."""
    import csv
    import hashlib
    from datetime import date

    if revision not in REVISIONS:
        raise LabelError(f"revision must be one of {REVISIONS[1:]}")
    previous = None
    if revision:
        earlier = REVISIONS[REVISIONS.index(revision) - 1]
        if not review_paths(version, root, earlier)["labels"].exists():
            raise LabelError(f"cannot freeze {revision}: label set {earlier or 'v1'} is not frozen")
        previous = (earlier, *load_labels(version, root, earlier))
    paths = review_paths(version, root, revision)
    if paths["labels"].exists() or paths["meta"].exists():
        raise LabelError(
            f"{paths['labels'].name} already exists: frozen labels are never overwritten"
        )
    with paths["csv"].open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    key = json.loads(paths["key"].read_text(encoding="utf-8"))
    problems = []
    if {r["row_id"] for r in rows} != set(key["rows"]):
        problems.append("row ids in the CSV do not match the key")
    for r in rows:
        label = r["human_label"].strip().lower()
        if r["row_id"] in key["rows"] and r["unit"] != key["rows"][r["row_id"]]["unit"]:
            problems.append(f"{r['row_id']}: unit changed from the generated file")
        if label not in ALLOWED.get(r["unit"], ()):
            problems.append(
                f"{r['row_id']}: {r['unit']} label {label!r} not in {ALLOWED.get(r['unit'])}"
            )
    if problems:
        raise LabelError(f"{len(problems)} problem(s):\n  " + "\n  ".join(problems))

    labels = []
    for r in sorted(rows, key=lambda r: r["row_id"]):
        k = key["rows"][r["row_id"]]
        labels.append(
            {
                "row_id": r["row_id"],
                "unit": r["unit"],
                "topic": r["topic"],
                "item_id": k["item_id"],
                "claim_index": k["claim_index"],
                "probe": k.get("probe", False),
                "donor_id": k.get("donor_id"),
                "human_label": r["human_label"].strip().lower(),
                "judge_label": k["judge_label"],
                "notes": r["notes"].strip(),
            }
        )
    with atomic_write(paths["labels"]) as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(paths["labels"].read_bytes()).hexdigest()
    meta = {
        "version": version,
        "created": date.today().isoformat(),
        "labeller": labeller,
        "labels": len(labels),
        "csv_sha256": hashlib.sha256(paths["csv"].read_bytes()).hexdigest(),
        "run_id": key["meta"].get("run_id"),
        "split": key["meta"].get("split"),
        "judge_model": key["meta"]["judge_model"],
        "judge_prompt": key["meta"]["judge_prompt"],
        "probes": key["meta"].get("probes"),
        "revision": revision or "v1",
        "sha256": digest,
    }
    if previous:
        earlier, old_labels, old_meta = previous
        old = {x["row_id"]: x for x in old_labels}
        meta["supersedes"] = {
            "revision": earlier or "v1",
            "file": review_paths(version, root, earlier)["labels"].name,
            "sha256": old_meta["sha256"],
        }
        meta["changed_rows"] = [
            x["row_id"]
            for x in labels
            if (x["human_label"], x["notes"])
            != (old[x["row_id"]]["human_label"], old[x["row_id"]]["notes"])
        ]
    with atomic_write(paths["meta"]) as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta

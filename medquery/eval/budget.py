"""Phase 4 judge budget: refuse to start a run that could push the phase's OpenAI judge spend
past the cap. Phase 4 runs are the ones whose generator runs on NVIDIA."""

import json
from dataclasses import dataclass
from pathlib import Path

PHASE4_CAP_USD = 3.00
HEADROOM = 1.5  # estimate x 1.5: answer_v2 answers may be longer, so judging costs more
FALLBACK_COST_PER_ITEM = 0.01  # used only before any judged run exists


@dataclass(frozen=True)
class BudgetCheck:
    spent: float  # judge cost already in Phase 4 runs (cached calls count too: conservative)
    estimate: float  # judge cost of the items still to run, with headroom
    cap: float

    @property
    def ok(self) -> bool:
        return self.spent + self.estimate <= self.cap

    def line(self) -> str:
        return (
            f"Phase 4 judge budget: ${self.spent:.2f} spent before this run; "
            f"${self.estimate:.2f} estimated for it (x{HEADROOM} headroom); cap ${self.cap:.2f}"
        )


def _runs(runs_dir: Path):
    for run_json in sorted(runs_dir.glob("*/run.json")):
        config = json.loads(run_json.read_text(encoding="utf-8"))["config"]
        items = run_json.parent / "items.jsonl"
        lines = items.read_text(encoding="utf-8").splitlines() if items.exists() else []
        yield run_json.parent.name, config, [json.loads(x) for x in lines if x.strip()]


def is_phase4(config: dict) -> bool:
    return config.get("generator_provider", "groq") == "nvidia"


def spent(runs_dir: Path, exclude: str = "") -> float:
    """Judge cost recorded in Phase 4 runs (all but `exclude`)."""
    return sum(
        r["cost_usd"]["judge"]
        for rid, config, records in _runs(runs_dir)
        if is_phase4(config) and rid != exclude
        for r in records
    )


def mean_judge_cost(runs_dir: Path) -> float:
    """Mean judge cost per judged item over every completed record so far."""
    costs = [
        r["cost_usd"]["judge"]
        for _, _, records in _runs(runs_dir)
        for r in records
        if r.get("judge")
    ]
    return sum(costs) / len(costs) if costs else FALLBACK_COST_PER_ITEM


def check(runs_dir: Path, run_id: str, todo: int, cap: float = PHASE4_CAP_USD) -> BudgetCheck:
    """`run_id`'s own finished items count as spent; its `todo` items are estimated."""
    already = spent(runs_dir, exclude=run_id) + sum(
        r["cost_usd"]["judge"]
        for rid, _, records in _runs(runs_dir)
        if rid == run_id
        for r in records
    )
    return BudgetCheck(already, todo * mean_judge_cost(runs_dir) * HEADROOM, cap)

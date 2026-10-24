"""Per-phase judge budget: refuse to start a run that could push the current phase's OpenAI
judge spend past the cap. Metered runs are the ones whose generator runs on NVIDIA (Phase 4 on);
runs of closed phases are listed by id, and every other metered run belongs to the current phase."""

import json
from dataclasses import dataclass
from pathlib import Path

CURRENT_PHASE = "Phase 5"
CAP_USD = 3.00  # per phase (Phase 4 and Phase 5 briefs)
CLOSED_PHASES = {
    "Phase 4": frozenset(
        {"ed1249bdc055", "4b3e118c95f5", "ff1371441790", "05a0d8c86909", "53088e55be95"}
    ),
}
HEADROOM = 1.5  # estimate x 1.5: answer_v2 answers may be longer, so judging costs more
FALLBACK_COST_PER_ITEM = 0.01  # used only before any judged run exists


@dataclass(frozen=True)
class BudgetCheck:
    spent: float  # judge cost already in this phase's runs (cached calls count: conservative)
    estimate: float  # judge cost of the items still to run, with headroom
    cap: float

    @property
    def ok(self) -> bool:
        return self.spent + self.estimate <= self.cap

    def line(self) -> str:
        return (
            f"{CURRENT_PHASE} judge budget: ${self.spent:.2f} spent before this run; "
            f"${self.estimate:.2f} estimated for it (x{HEADROOM} headroom); cap ${self.cap:.2f}"
        )


def _runs(runs_dir: Path):
    for run_json in sorted(runs_dir.glob("*/run.json")):
        config = json.loads(run_json.read_text(encoding="utf-8"))["config"]
        items = run_json.parent / "items.jsonl"
        lines = items.read_text(encoding="utf-8").splitlines() if items.exists() else []
        yield run_json.parent.name, config, [json.loads(x) for x in lines if x.strip()]


def is_metered(config: dict) -> bool:
    return config.get("generator_provider", "groq") == "nvidia"


def phase_of(run_id: str, config: dict) -> str | None:
    if not is_metered(config):
        return None
    closed = [name for name, ids in CLOSED_PHASES.items() if run_id in ids]
    return closed[0] if closed else CURRENT_PHASE


def spent(runs_dir: Path, exclude: str = "", phase: str = CURRENT_PHASE) -> float:
    """Judge cost recorded in a phase's runs (all but `exclude`)."""
    return sum(
        r["cost_usd"]["judge"]
        for rid, config, records in _runs(runs_dir)
        if phase_of(rid, config) == phase and rid != exclude
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


def check(runs_dir: Path, run_id: str, todo: int, cap: float = CAP_USD) -> BudgetCheck:
    """`run_id`'s own finished items count as spent; its `todo` items are estimated."""
    already = spent(runs_dir, exclude=run_id) + sum(
        r["cost_usd"]["judge"]
        for rid, _, records in _runs(runs_dir)
        if rid == run_id
        for r in records
    )
    return BudgetCheck(already, todo * mean_judge_cost(runs_dir) * HEADROOM, cap)

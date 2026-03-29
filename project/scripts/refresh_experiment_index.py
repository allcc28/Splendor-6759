"""
Build a single source of truth for experiment artifacts.

Outputs:
  - project/experiments/reports/EXPERIMENT_INDEX.json
  - project/experiments/reports/EXPERIMENT_INDEX.md
  - project/experiments/evaluation/LATEST.md

The index is intentionally non-destructive: it does not move or rename files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "project" / "logs"
EVAL_DIR = ROOT / "project" / "experiments" / "evaluation"
ROBUST_DIR = EVAL_DIR / "robust"
REPORTS_DIR = ROOT / "project" / "experiments" / "reports"
LATEST_MD_PATH = EVAL_DIR / "LATEST.md"


@dataclass
class RunRecord:
    run_name: str
    run_path: str
    final_model: str | None
    best_model: str | None
    config: str | None
    last_modified: str


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(ROOT)).replace("\\", "/")


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def paired_report(path: Path) -> Path | None:
    report_candidates = [
        path.with_name(f"{path.stem}_report.md"),
        path.with_suffix(".md"),
    ]
    for candidate in report_candidates:
        if candidate.exists():
            return candidate
    return None


def collect_log_runs() -> list[RunRecord]:
    records: list[RunRecord] = []
    if not LOGS_DIR.exists():
        return records

    excluded_dirs = {"archived_logs", "archive_incomplete"}

    for run_dir in sorted(LOGS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        if run_dir.name in excluded_dirs:
            continue

        final_model = run_dir / "final_model.zip"
        best_model = run_dir / "eval" / "best_model.zip"
        config = run_dir / "config.yaml"

        records.append(
            RunRecord(
                run_name=run_dir.name,
                run_path=relpath(run_dir),
                final_model=relpath(final_model) if final_model.exists() else None,
                best_model=relpath(best_model) if best_model.exists() else None,
                config=relpath(config) if config.exists() else None,
                last_modified=iso(run_dir.stat().st_mtime),
            )
        )

    return records


def robust_eval_score(entry: dict[str, Any]) -> float:
    keys = ["Random (wrapper)", "RandomAgent", "GreedyAgent"]
    values: list[float] = []
    for key in keys:
        block = entry.get("results", {}).get(key, {})
        value = block.get("win_rate_pooled")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else -1.0


def detect_robust_family(path: Path, eval_tag: str, model_path: str, event_shaping_enabled: bool) -> str:
    parts = {part.lower() for part in path.parts}
    if "score_based" in parts:
        return "score_based"
    if "event_based" in parts:
        return "event_based"

    tag = (eval_tag or "").lower()
    model = (model_path or "").lower()

    if event_shaping_enabled or "event" in tag or "event" in model:
        return "event_based"
    if "score" in tag or "score" in model:
        return "score_based"
    return "other"


def collect_robust_evals() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not ROBUST_DIR.exists():
        return items

    for path in sorted(ROBUST_DIR.rglob("robust_eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = payload.get("_meta", {})
        model_path = meta.get("model_path", "")
        family = detect_robust_family(
            path=path,
            eval_tag=meta.get("eval_tag", ""),
            model_path=model_path,
            event_shaping_enabled=bool(meta.get("event_shaping_enabled", False)),
        )
        report_path = paired_report(path)
        item = {
            "file": relpath(path),
            "report_file": relpath(report_path),
            "timestamp": meta.get("timestamp", ""),
            "eval_tag": meta.get("eval_tag", ""),
            "games_per_opponent": meta.get("games_per_opponent"),
            "model_path": model_path,
            "family": family,
            "results": {
                "Random (wrapper)": {
                    "win_rate_pooled": payload.get("Random (wrapper)", {}).get("win_rate_pooled"),
                    "wilson_ci": [
                        payload.get("Random (wrapper)", {}).get("wilson_ci_95_lo"),
                        payload.get("Random (wrapper)", {}).get("wilson_ci_95_hi"),
                    ],
                },
                "RandomAgent": {
                    "win_rate_pooled": payload.get("RandomAgent", {}).get("win_rate_pooled"),
                    "wilson_ci": [
                        payload.get("RandomAgent", {}).get("wilson_ci_95_lo"),
                        payload.get("RandomAgent", {}).get("wilson_ci_95_hi"),
                    ],
                },
                "GreedyAgent": {
                    "win_rate_pooled": payload.get("GreedyAgent", {}).get("win_rate_pooled"),
                    "wilson_ci": [
                        payload.get("GreedyAgent", {}).get("wilson_ci_95_lo"),
                        payload.get("GreedyAgent", {}).get("wilson_ci_95_hi"),
                    ],
                },
            },
        }
        item["composite_score"] = robust_eval_score(item)
        items.append(item)

    items.sort(key=lambda entry: entry.get("composite_score", -1.0), reverse=True)
    return items


def collect_mcts_benchmarks() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not ROBUST_DIR.exists():
        return items

    for path in sorted(ROBUST_DIR.rglob("mcts_benchmark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = payload.get("meta", {})
        parent = path.parent.name.lower()
        bucket = parent if parent in {"canonical", "archive"} else "other"
        report_path = paired_report(path)
        items.append(
            {
                "file": relpath(path),
                "report_file": relpath(report_path),
                "timestamp": meta.get("timestamp", ""),
                "games_per_matchup": meta.get("games_per_matchup"),
                "iterations": meta.get("iterations"),
                "ppo_model": meta.get("ppo_model"),
                "bucket": bucket,
                "rows": len(payload.get("results", [])),
            }
        )

    return items


def best_by_family(robust_entries: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    out: dict[str, dict[str, Any] | None] = {}
    for family in ["score_based", "event_based", "other"]:
        out[family] = next((entry for entry in robust_entries if entry.get("family") == family), None)
    return out


def preferred_path(entry: dict[str, Any] | None) -> str | None:
    if not entry:
        return None
    return entry.get("report_file") or entry.get("file")


def run_is_usable(run: RunRecord) -> bool:
    """A run is usable for continuation/evaluation if it has at least one model artifact."""
    return bool(run.final_model or run.best_model)


def pick_latest_training_run(runs: list[RunRecord]) -> RunRecord | None:
    """Prefer newest usable run; fall back to newest run if none are usable."""
    if not runs:
        return None

    for run in runs:
        if run_is_usable(run):
            return run

    return runs[0]


def build_payload() -> dict[str, Any]:
    runs = collect_log_runs()
    robust = collect_robust_evals()
    mcts = collect_mcts_benchmarks()

    family_best = best_by_family(robust)
    canonical_mcts = next((entry for entry in mcts if entry.get("bucket") == "canonical"), None)
    latest_run = pick_latest_training_run(runs)
    best_overall = robust[0] if robust else None

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "recommended": {
            "latest_training_run": asdict(latest_run) if latest_run else None,
            "best_robust_model": best_overall.get("model_path") if best_overall else None,
            "best_robust_eval": preferred_path(best_overall),
            "best_composite_score": best_overall.get("composite_score") if best_overall else None,
            "best_score_based_eval": preferred_path(family_best["score_based"]),
            "best_event_based_eval": preferred_path(family_best["event_based"]),
            "canonical_mcts_eval": preferred_path(canonical_mcts),
        },
        "log_runs": [asdict(run) for run in runs],
        "robust_evaluations": robust,
        "mcts_benchmarks": mcts,
    }


def as_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "NA"
    return f"{float(value):.1f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    rec = payload.get("recommended", {})
    lines: list[str] = []
    lines.append("# Experiment Index")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Total log runs: {len(payload['log_runs'])}")
    lines.append(f"- Robust eval files: {len(payload['robust_evaluations'])}")
    lines.append(f"- MCTS benchmark files: {len(payload['mcts_benchmarks'])}")
    lines.append("")
    lines.append("## Quick Entry Points")
    lines.append("")
    lines.append(f"- Latest training run: {rec.get('latest_training_run', {}).get('run_path') if rec.get('latest_training_run') else 'NA'}")
    lines.append(f"- Best score-based robust report: {rec.get('best_score_based_eval') or 'NA'}")
    lines.append(f"- Best event-based robust report: {rec.get('best_event_based_eval') or 'NA'}")
    lines.append(f"- Canonical MCTS benchmark: {rec.get('canonical_mcts_eval') or 'NA'}")
    lines.append("")
    lines.append("## Best Overall Robust Model")
    lines.append("")
    lines.append(f"- Model path: {rec.get('best_robust_model') or 'NA'}")
    lines.append(f"- Evidence file: {rec.get('best_robust_eval') or 'NA'}")
    score = rec.get("best_composite_score")
    if isinstance(score, (int, float)):
        lines.append(f"- Composite score (mean win% vs Random/RandomAgent/Greedy): {score:.2f}")
    else:
        lines.append("- Composite score: NA")
    lines.append("")
    lines.append("## Robust Leaderboard")
    lines.append("")
    lines.append("| Rank | Family | Eval Tag | Model Path | Rnd(wrapper) | RandomAgent | Greedy | Composite | Report |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---|")
    for idx, entry in enumerate(payload.get("robust_evaluations", [])[:10], start=1):
        results = entry.get("results", {})
        lines.append(
            "| {idx} | {family} | {tag} | {model} | {r0} | {r1} | {r2} | {comp:.2f} | {report} |".format(
                idx=idx,
                family=entry.get("family", "other"),
                tag=entry.get("eval_tag", ""),
                model=entry.get("model_path", ""),
                r0=as_pct(results.get("Random (wrapper)", {}).get("win_rate_pooled")),
                r1=as_pct(results.get("RandomAgent", {}).get("win_rate_pooled")),
                r2=as_pct(results.get("GreedyAgent", {}).get("win_rate_pooled")),
                comp=float(entry.get("composite_score", -1.0)),
                report=preferred_path(entry) or "NA",
            )
        )
    lines.append("")
    lines.append("## Latest 10 Training Runs")
    lines.append("")
    lines.append("| Run | Last Modified | best_model.zip | final_model.zip | config.yaml |")
    lines.append("|---|---|---|---|---|")
    for run in payload.get("log_runs", [])[:10]:
        lines.append(
            "| {run} | {ts} | {best} | {final} | {cfg} |".format(
                run=run.get("run_name", ""),
                ts=run.get("last_modified", ""),
                best=run.get("best_model") or "-",
                final=run.get("final_model") or "-",
                cfg=run.get("config") or "-",
            )
        )
    lines.append("")
    lines.append("## Latest 10 MCTS Benchmarks")
    lines.append("")
    lines.append("| Bucket | File | Timestamp | Games/Matchup | Iterations | PPO Model |")
    lines.append("|---|---|---|---:|---|---|")
    for item in payload.get("mcts_benchmarks", [])[:10]:
        lines.append(
            "| {bucket} | {file} | {ts} | {games} | {iterations} | {ppo} |".format(
                bucket=item.get("bucket", "other"),
                file=preferred_path(item) or item.get("file", ""),
                ts=item.get("timestamp", ""),
                games=item.get("games_per_matchup", ""),
                iterations=item.get("iterations", ""),
                ppo=item.get("ppo_model", ""),
            )
        )
    lines.append("")
    lines.append("## How To Refresh")
    lines.append("")
    lines.append("Run: python project/scripts/refresh_experiment_index.py")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_latest_markdown(payload: dict[str, Any]) -> str:
    rec = payload.get("recommended", {})
    latest_run = rec.get("latest_training_run", {})
    lines: list[str] = []
    lines.append("# Evaluation Entry Point")
    lines.append("")
    lines.append("Use this file when you want the latest key results quickly.")
    lines.append("")
    lines.append(f"- Last refreshed: {payload['generated_at']}")
    lines.append("")
    lines.append("## PPO (Robust)")
    lines.append("")
    lines.append(f"- Best score-based report: {rec.get('best_score_based_eval') or 'NA'}")
    lines.append(f"- Best event-based report: {rec.get('best_event_based_eval') or 'NA'}")
    lines.append(f"- Latest training run: {latest_run.get('run_path') or 'NA'}")
    lines.append("")
    lines.append("## MCTS Benchmarks")
    lines.append("")
    lines.append(f"- Canonical benchmark: {rec.get('canonical_mcts_eval') or 'NA'}")
    lines.append("- Archive/debug bucket: project/experiments/evaluation/robust/mcts/archive/")
    lines.append("")
    lines.append("## Global Index")
    lines.append("")
    lines.append("- project/experiments/reports/EXPERIMENT_INDEX.md")
    lines.append("")
    lines.append("## Legacy Warning")
    lines.append("")
    lines.append("- Legacy Phase 6 score-based outputs are archived at project/experiments/evaluation/archive/legacy_phase6/.")
    lines.append("- Do not use legacy archive outputs for current benchmark conclusions.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    payload = build_payload()

    json_path = REPORTS_DIR / "EXPERIMENT_INDEX.json"
    md_path = REPORTS_DIR / "EXPERIMENT_INDEX.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    LATEST_MD_PATH.write_text(render_latest_markdown(payload), encoding="utf-8")

    print(f"Wrote {relpath(json_path)}")
    print(f"Wrote {relpath(md_path)}")
    print(f"Wrote {relpath(LATEST_MD_PATH)}")


if __name__ == "__main__":
    main()

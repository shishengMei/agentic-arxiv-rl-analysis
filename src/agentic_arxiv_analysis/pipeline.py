"""Import and analyse AgenticArXiv-RL experiment summaries.

Only aggregate metrics are copied into this project. Raw prompts, model outputs, and
upstream source code stay in the source checkout and are referenced by path + commit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UPSTREAM_URL = "https://github.com/Algorineko/AgenticArXiv-RL"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    stage: str
    split: str
    path: str


RUN_SPECS = (
    RunSpec("base_dev", "Base", "dev", "artifacts/qwen25_15b_base_v2_62/dev/summary.json"),
    RunSpec(
        "base_iid_test", "Base", "iid_test", "artifacts/qwen25_15b_base_v2_62/iid_test/summary.json"
    ),
    RunSpec(
        "base_ood_test", "Base", "ood_test", "artifacts/qwen25_15b_base_v2_62/ood_test/summary.json"
    ),
    RunSpec(
        "sft_train",
        "SFT",
        "train",
        "artifacts/qwen25_15b_sft_e1_final_train_r3_semantic/summary.json",
    ),
    RunSpec(
        "sft_dev",
        "SFT",
        "dev",
        "artifacts/qwen25_15b_sft_e1_final_dev_r3_semantic/summary.json",
    ),
    RunSpec("sft_v5_dev", "SFT", "v5_dev", "artifacts/eval_sft_v5_dev_r3_s45/summary.json"),
    RunSpec("grpo_v5_dev", "GRPO", "v5_dev", "artifacts/eval_grpo_v5_dev_r3_s45/summary.json"),
    RunSpec(
        "sft_v5_rl_train",
        "SFT",
        "v5_rl_train",
        "artifacts/eval_sft_v5_rl_train_r3_s45/summary.json",
    ),
    RunSpec(
        "grpo_v5_rl_train",
        "GRPO",
        "v5_rl_train",
        "artifacts/eval_grpo_v5_rl_train_r3_s45/summary.json",
    ),
)

METRICS = (
    "completion_rate",
    "strict_success_rate",
    "tool_accuracy",
    "arg_accuracy",
    "ref_accuracy",
    "false_finish_rate",
    "avg_iterations",
    "avg_tokens",
    "avg_total_ms",
    "avg_tool_failures",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _extract_run(repo: Path, spec: RunSpec) -> dict[str, Any]:
    path = repo / spec.path
    summary = _read_json(path)
    agents = summary.get("summary_by_agent", {})
    if len(agents) != 1:
        raise ValueError(f"{spec.path}: expected one evaluated agent, found {len(agents)}")
    agent_name, metrics = next(iter(agents.items()))
    missing = [key for key in METRICS if key not in metrics]
    if missing:
        raise ValueError(f"{spec.path}: missing aggregate metrics: {', '.join(missing)}")
    return {
        "run_id": spec.run_id,
        "stage": spec.stage,
        "split": spec.split,
        "source_path": spec.path,
        "source_sha256": _sha256(path),
        "model": summary.get("model", ""),
        "agent": agent_name,
        "sample_count": summary["sample_count"],
        **{key: metrics[key] for key in METRICS},
    }


def build_dataset(source_repo: Path) -> dict[str, Any]:
    """Build a compact, attributable dataset from an upstream checkout."""

    source_repo = source_repo.resolve()
    manifest_path = source_repo / "data/splits/v5_grpo_train.json"
    manifest = _read_json(manifest_path)
    formal = manifest["formal_run"]
    evaluation = manifest["evaluation"]
    dataset = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": UPSTREAM_URL,
            "commit": _git_commit(source_repo),
            "manifest_path": "data/splits/v5_grpo_train.json",
            "manifest_sha256": _sha256(manifest_path),
        },
        "protocol": evaluation["protocol"],
        "training": {
            "requested_steps": formal["requested_max_steps"],
            "actual_steps": formal["actual_steps"],
            "runtime_seconds": formal["runtime_seconds"],
            "peak_vram_gib": formal["peak_vram_gib"],
            "rollout_count": formal["rollout_count"],
            "prompt_group_count": formal["prompt_group_count"],
            "informative_group_fraction": formal["informative_group_fraction"],
            "stop_trigger": formal["stop_trigger"],
        },
        "evaluation_claim": {
            "decision": evaluation["decision"],
            "reward_hacking_check": evaluation["reward_hacking_check"],
            "improved_dev_task": evaluation["dev"]["improved_task"],
        },
        "runs": [_extract_run(source_repo, spec) for spec in RUN_SPECS],
    }
    validate_dataset(dataset, evaluation=evaluation)
    return dataset


def _run_index(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {run["run_id"]: run for run in dataset["runs"]}


def validate_dataset(dataset: dict[str, Any], *, evaluation: dict[str, Any] | None = None) -> None:
    """Fail loudly when provenance or controlled comparisons are inconsistent."""

    if dataset.get("schema_version") != 1:
        raise ValueError("unsupported dataset schema")
    runs = dataset.get("runs", [])
    if len(runs) != len(RUN_SPECS):
        raise ValueError(f"expected {len(RUN_SPECS)} runs, found {len(runs)}")
    ids = [run["run_id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("run_id values must be unique")
    for run in runs:
        if run["sample_count"] <= 0:
            raise ValueError(f"{run['run_id']}: sample_count must be positive")
        for metric in ("completion_rate", "strict_success_rate", "tool_accuracy", "arg_accuracy"):
            if not 0.0 <= run[metric] <= 1.0:
                raise ValueError(f"{run['run_id']}: {metric} is outside [0, 1]")

    if evaluation is not None:
        index = _run_index(dataset)
        expected = {
            "sft_v5_rl_train": evaluation["rl_train"]["sft_strict_success_rate"],
            "grpo_v5_rl_train": evaluation["rl_train"]["grpo_strict_success_rate"],
            "sft_v5_dev": evaluation["dev"]["sft_strict_success_rate"],
            "grpo_v5_dev": evaluation["dev"]["grpo_strict_success_rate"],
        }
        for run_id, rate in expected.items():
            if abs(index[run_id]["strict_success_rate"] - rate) > 1e-12:
                raise ValueError(f"{run_id}: summary disagrees with the experiment manifest")


def comparisons(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only apples-to-apples SFT vs GRPO comparisons."""

    index = _run_index(dataset)
    pairs = (
        ("v5_rl_train", "sft_v5_rl_train", "grpo_v5_rl_train"),
        ("v5_dev", "sft_v5_dev", "grpo_v5_dev"),
    )
    rows = []
    for split, before_id, after_id in pairs:
        before, after = index[before_id], index[after_id]
        if before["sample_count"] != after["sample_count"]:
            raise ValueError(f"{split}: paired runs have different sample counts")
        rows.append(
            {
                "split": split,
                "sample_count": before["sample_count"],
                "sft_strict_success_rate": before["strict_success_rate"],
                "grpo_strict_success_rate": after["strict_success_rate"],
                "strict_success_delta_pp": 100
                * (after["strict_success_rate"] - before["strict_success_rate"]),
                "tool_accuracy_delta_pp": 100 * (after["tool_accuracy"] - before["tool_accuracy"]),
                "argument_accuracy_delta_pp": 100
                * (after["arg_accuracy"] - before["arg_accuracy"]),
                "avg_tool_failures_delta": after["avg_tool_failures"] - before["avg_tool_failures"],
                "avg_tokens_delta": after["avg_tokens"] - before["avg_tokens"],
            }
        )
    return rows


def _write_metrics_csv(dataset: dict[str, Any], output_path: Path) -> None:
    fields = ("run_id", "stage", "split", "sample_count", *METRICS)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dataset["runs"])


def _write_comparisons_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _write_report(dataset: dict[str, Any], rows: list[dict[str, Any]], output_path: Path) -> None:
    index = _run_index(dataset)
    train, dev = rows
    training = dataset["training"]
    claim = dataset["evaluation_claim"]
    lines = [
        "# AgenticArXiv-RL 实验复现报告",
        "",
        f"- 上游提交：`{dataset['source']['commit']}`",
        (
            f"- 评测协议：seed={dataset['protocol']['seed']}，"
            f"repeat={dataset['protocol']['repeat']}，后端={dataset['protocol']['backend']}，"
            f"工具执行={dataset['protocol']['tool_execution']}"
        ),
        (
            f"- GRPO 训练：{training['actual_steps']}/{training['requested_steps']} steps，"
            f"{training['runtime_seconds']:.1f}s，峰值显存 "
            f"{training['peak_vram_gib']:.2f} GiB"
        ),
        "",
        "## 核心结果",
        "",
        "| 同协议数据集 | 样本数 | SFT 严格成功率 | GRPO 严格成功率 | 变化 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| RL train diagnostic | {train['sample_count']} | "
            f"{_pct(train['sft_strict_success_rate'])} | "
            f"{_pct(train['grpo_strict_success_rate'])} | "
            f"+{train['strict_success_delta_pp']:.2f} pp |"
        ),
        (
            f"| Dev | {dev['sample_count']} | {_pct(dev['sft_strict_success_rate'])} | "
            f"{_pct(dev['grpo_strict_success_rate'])} | "
            f"+{dev['strict_success_delta_pp']:.2f} pp |"
        ),
        "",
        "## 诊断",
        "",
        (
            f"- Dev 工具准确率保持 {_pct(index['grpo_v5_dev']['tool_accuracy'])}，"
            f"参数准确率从 {_pct(index['sft_v5_dev']['arg_accuracy'])} 升至 "
            f"{_pct(index['grpo_v5_dev']['arg_accuracy'])}。"
        ),
        (
            f"- RL train diagnostic 的平均 token 从 "
            f"{index['sft_v5_rl_train']['avg_tokens']:.1f} 降至 "
            f"{index['grpo_v5_rl_train']['avg_tokens']:.1f}"
            f"（{train['avg_tokens_delta']:.1f}）。"
        ),
        f"- Dev 增益来自任务 `{claim['improved_dev_task']}`，不能外推为广泛泛化提升。",
        f"- 训练提前停止原因：{training['stop_trigger']}",
        "",
        "## 结论边界",
        "",
        f"> {claim['decision']}",
        "",
        f"> {claim['reward_hacking_check']}",
        "",
        (
            "Base/SFT 的历史结果保留在 `metrics.csv` 中用于背景观察，但只有 "
            "seed、repeat、split 和评测协议完全一致的 v5 SFT/GRPO 对被用于因果式前后比较。"
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_svg(rows: list[dict[str, Any]], output_path: Path) -> None:
    width, height = 760, 390
    chart_left, chart_bottom, chart_height = 100, 315, 240
    groups = (("RL train", rows[0]), ("Dev", rows[1]))
    colors = {"SFT": "#64748b", "GRPO": "#2563eb"}
    bars: list[str] = []
    for group_index, (label, row) in enumerate(groups):
        group_x = 180 + group_index * 300
        for model_index, model in enumerate(("SFT", "GRPO")):
            value = row[f"{model.lower()}_strict_success_rate"]
            bar_height = value * chart_height
            x = group_x + model_index * 75
            y = chart_bottom - bar_height
            bars.extend(
                [
                    (
                        f'<rect x="{x}" y="{y:.1f}" width="54" '
                        f'height="{bar_height:.1f}" fill="{colors[model]}" rx="4"/>'
                    ),
                    (
                        f'<text x="{x + 27}" y="{y - 8:.1f}" text-anchor="middle" '
                        f'class="value">{100 * value:.1f}%</text>'
                    ),
                    (
                        f'<text x="{x + 27}" y="{chart_bottom + 22}" '
                        f'text-anchor="middle" class="small">{model}</text>'
                    ),
                ]
            )
        bars.append(
            f'<text x="{group_x + 64}" y="{chart_bottom + 52}" text-anchor="middle" '
            f'class="label">{label}</text>'
        )
    grid = []
    for pct in (0, 25, 50, 75, 100):
        y = chart_bottom - pct / 100 * chart_height
        grid.append(f'<line x1="{chart_left}" y1="{y}" x2="700" y2="{y}" class="grid"/>')
        grid.append(f'<text x="88" y="{y + 5}" text-anchor="end" class="small">{pct}%</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 20px system-ui, sans-serif; fill: #0f172a; }}
  .label {{ font: 600 14px system-ui, sans-serif; fill: #334155; }}
  .value {{ font: 700 13px system-ui, sans-serif; fill: #0f172a; }}
  .small {{ font: 12px system-ui, sans-serif; fill: #475569; }}
  .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="36" y="38" class="title">SFT vs GRPO strict success rate</text>
{"".join(grid)}
{"".join(bars)}
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")


def generate_outputs(dataset_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    dataset = _read_json(dataset_path)
    validate_dataset(dataset)
    rows = comparisons(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(dataset, output_dir / "metrics.csv")
    _write_comparisons_csv(rows, output_dir / "comparisons.csv")
    _write_report(dataset, rows, output_dir / "report.md")
    _write_svg(rows, output_dir / "strict_success.svg")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    import_parser = subparsers.add_parser("import", help="extract aggregate upstream metrics")
    import_parser.add_argument("--source-repo", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, default=Path("data/experiment.json"))
    report_parser = subparsers.add_parser("report", help="validate data and generate results")
    report_parser.add_argument("--data", type=Path, default=Path("data/experiment.json"))
    report_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "import":
        dataset = build_dataset(args.source_repo)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}")
    else:
        rows = generate_outputs(args.data, args.output_dir)
        for row in rows:
            print(f"{row['split']}: strict success {row['strict_success_delta_pp']:+.2f} pp")


if __name__ == "__main__":
    main()

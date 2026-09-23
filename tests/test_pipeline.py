from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentic_arxiv_analysis.pipeline import comparisons, generate_outputs, validate_dataset


def _run(run_id: str, stage: str, split: str, strict: float, *, sample_count: int) -> dict:
    return {
        "run_id": run_id,
        "stage": stage,
        "split": split,
        "source_path": f"artifacts/{run_id}/summary.json",
        "source_sha256": "0" * 64,
        "model": run_id,
        "agent": "regex",
        "sample_count": sample_count,
        "completion_rate": 1.0,
        "strict_success_rate": strict,
        "tool_accuracy": 0.8,
        "arg_accuracy": 0.6,
        "ref_accuracy": 0.7,
        "false_finish_rate": 0.0,
        "avg_iterations": 2.0,
        "avg_tokens": 3000.0,
        "avg_total_ms": 1800.0,
        "avg_tool_failures": 0.2,
    }


def _dataset() -> dict:
    runs = [
        _run("base_dev", "Base", "dev", 0.2, sample_count=24),
        _run("base_iid_test", "Base", "iid_test", 0.3, sample_count=42),
        _run("base_ood_test", "Base", "ood_test", 0.0, sample_count=12),
        _run("sft_train", "SFT", "train", 0.7, sample_count=108),
        _run("sft_dev", "SFT", "dev", 0.4, sample_count=24),
        _run("sft_v5_dev", "SFT", "v5_dev", 10 / 24, sample_count=24),
        _run("grpo_v5_dev", "GRPO", "v5_dev", 12 / 24, sample_count=24),
        _run("sft_v5_rl_train", "SFT", "v5_rl_train", 1 / 21, sample_count=21),
        _run("grpo_v5_rl_train", "GRPO", "v5_rl_train", 7 / 21, sample_count=21),
    ]
    return {
        "schema_version": 1,
        "source": {"commit": "abc123"},
        "protocol": {
            "seed": 45,
            "repeat": 3,
            "backend": "transformers",
            "tool_execution": "offline_snapshot_replay",
        },
        "training": {
            "requested_steps": 120,
            "actual_steps": 47,
            "runtime_seconds": 515.1,
            "peak_vram_gib": 8.75,
            "stop_trigger": "reward variance saturation",
        },
        "evaluation_claim": {
            "decision": "Narrow targeted gain.",
            "reward_hacking_check": "No observed reward hacking.",
            "improved_dev_task": "search_kw_agentic_rl",
        },
        "runs": runs,
    }


class PipelineTests(unittest.TestCase):
    def test_controlled_deltas_are_percentage_points(self) -> None:
        rows = comparisons(_dataset())
        self.assertAlmostEqual(rows[0]["strict_success_delta_pp"], 28.5714285714)
        self.assertAlmostEqual(rows[1]["strict_success_delta_pp"], 8.3333333333)

    def test_mismatched_pair_size_is_rejected(self) -> None:
        dataset = _dataset()
        dataset["runs"][-1]["sample_count"] = 20
        with self.assertRaisesRegex(ValueError, "different sample counts"):
            comparisons(dataset)

    def test_invalid_rate_is_rejected(self) -> None:
        dataset = _dataset()
        dataset["runs"][0]["strict_success_rate"] = 1.01
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_dataset(dataset)

    def test_report_pipeline_writes_all_artifacts(self) -> None:
        dataset = _dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "experiment.json"
            data_path.write_text(json.dumps(dataset), encoding="utf-8")
            generate_outputs(data_path, root / "results")
            for filename in ("metrics.csv", "comparisons.csv", "report.md", "strict_success.svg"):
                self.assertTrue((root / "results" / filename).is_file())
            report = (root / "results" / "report.md").read_text(encoding="utf-8")
            self.assertIn("+28.57 pp", report)
            self.assertIn("+8.33 pp", report)

    def test_committed_experiment_is_auditable(self) -> None:
        dataset = json.loads((PROJECT_ROOT / "data/experiment.json").read_text(encoding="utf-8"))
        validate_dataset(dataset)
        rows = comparisons(dataset)
        self.assertEqual(dataset["source"]["commit"], "50b1ef1c345f79d3ca8e3ce1ecbc953a4f1d71b0")
        self.assertEqual([row["sample_count"] for row in rows], [21, 24])
        self.assertAlmostEqual(rows[0]["strict_success_delta_pp"], 28.5714285714)
        self.assertAlmostEqual(rows[1]["strict_success_delta_pp"], 8.3333333333)


if __name__ == "__main__":
    unittest.main()

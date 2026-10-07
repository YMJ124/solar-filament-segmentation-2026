"""Create and verify an immutable snapshot of a scored model baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="artifacts/runs/global_resnet34_fold0",
        type=Path,
    )
    parser.add_argument(
        "--config",
        default="configs/global_resnet34_fold0.yaml",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/baselines/model_a_fold0_epoch030",
        type=Path,
    )
    parser.add_argument("--public-score", default=0.07, type=float)
    parser.add_argument(
        "--score-source",
        default="user_provided_next_stage_plan",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing frozen snapshot without modifying it.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def tree_fingerprint(paths: Iterable[Path], project_root: Path) -> dict[str, Any]:
    files = sorted({path.resolve() for path in paths if path.is_file()})
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        rel = relative(path, project_root)
        size = path.stat().st_size
        file_hash = sha256_file(path)
        digest.update(f"{rel}\0{size}\0{file_hash}\n".encode())
        total_bytes += size
    return {
        "algorithm": "sha256(path\\0size\\0file_sha256\\n)",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def artifact_record(
    source: Path,
    frozen: Path,
    project_root: Path,
    frozen_display_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "source_path": relative(source, project_root),
        "frozen_path": relative(frozen_display_path or frozen, project_root),
        "bytes": frozen.stat().st_size,
        "sha256": sha256_file(frozen),
    }


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("PyTorch is required to inspect the baseline checkpoint") from error

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return {
        "schema_version": checkpoint.get("schema_version"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "best_metric": checkpoint.get("best_metric"),
        "best_metric_name": checkpoint.get("best_metric_name"),
        "model_kind": checkpoint.get("model_kind"),
        "config_fingerprint": checkpoint.get("config_fingerprint"),
    }


def git_state(project_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "available": False,
            "commit": None,
            "reason": "project directory is not a Git worktree",
        }
    return {"available": True, "commit": result.stdout.strip(), "reason": None}


def required_source_files(project_root: Path, run_dir: Path, config: Path) -> list[Path]:
    fixed = [
        config,
        run_dir / "config.resolved.yaml",
        run_dir / "run_manifest.json",
        run_dir / "checkpoints/best.pt",
        run_dir / "metrics/epochs.jsonl",
        run_dir / "metrics/full_pq_epoch030.json",
        run_dir / "submission_modelA_fold0_best.csv",
        run_dir / "submission_modelA_fold0_generation_audit.json",
        run_dir / "submission_modelA_fold0_reaudit.json",
        run_dir / "submission_console.log",
        run_dir / "train_console.log",
        project_root / "requirements.txt",
        project_root / "pytest.ini",
    ]
    dynamic = [
        *sorted((run_dir / "oof_best").glob("*.npz")),
        run_dir / "oof_best/manifest.json",
        *sorted((project_root / "src").rglob("*.py")),
        *sorted((project_root / "scripts").glob("*.py")),
        *sorted((project_root / "tests").glob("*.py")),
    ]
    required = fixed + dynamic
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required baseline files: {missing}")
    return sorted(set(required))


def copy_payload(files: Iterable[Path], project_root: Path, payload_root: Path) -> None:
    for source in files:
        destination = payload_root / relative(source, project_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def load_epoch_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def validate_baseline(
    run_dir: Path,
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
    oof_manifest: dict[str, Any],
    generation_audit: dict[str, Any],
    reaudit: dict[str, Any],
) -> None:
    if checkpoint["epoch"] != 30:
        raise ValueError(f"expected best checkpoint epoch 30, got {checkpoint['epoch']}")
    if checkpoint["best_metric_name"] != "official_pq":
        raise ValueError("best checkpoint was not selected by official_pq")
    if checkpoint["best_metric"] != metrics["official_pq"]:
        raise ValueError("checkpoint metric does not match epoch-30 full-image report")
    if metrics["official_pq"] != oof_manifest["official_pq"]:
        raise ValueError("epoch-30 report and OOF manifest disagree")
    oof_files = sorted((run_dir / "oof_best").glob("*.npz"))
    if len(oof_files) != metrics["physical_images"]:
        raise ValueError("OOF file count does not match evaluated physical images")
    if sorted(path.name for path in oof_files) != sorted(metrics["oof_probability_files"]):
        raise ValueError("OOF filenames do not match the best metric report")
    summary = generation_audit["summary"]
    if summary["rows"] != reaudit["rows"] or not reaudit["rles_validated"]:
        raise ValueError("submission generation audit and independent reaudit disagree")
    if reaudit["expected_images"] != 180:
        raise ValueError("submission audit does not cover the expected test set")


def payload_manifest(payload_root: Path, project_root: Path) -> dict[str, Any]:
    records = []
    digest = hashlib.sha256()
    for path in sorted(item for item in payload_root.rglob("*") if item.is_file()):
        rel_payload = path.relative_to(payload_root).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        records.append({"path": rel_payload, "bytes": size, "sha256": file_hash})
        digest.update(f"{rel_payload}\0{size}\0{file_hash}\n".encode())
    return {
        "schema_version": 1,
        "algorithm": "sha256(path\\0size\\0file_sha256\\n)",
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "payload_sha256": digest.hexdigest(),
        "files": records,
        "project_root_at_freeze": str(project_root.resolve()),
    }


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def verify_snapshot(output_dir: Path) -> None:
    manifest = json_load(output_dir / "archive_manifest.json")
    payload_root = output_dir / "payload"
    errors = []
    digest = hashlib.sha256()
    for item in manifest["files"]:
        path = payload_root / item["path"]
        if not path.is_file():
            errors.append(f"missing: {item['path']}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != item["bytes"] or actual_hash != item["sha256"]:
            errors.append(f"mismatch: {item['path']}")
        digest.update(f"{item['path']}\0{actual_size}\0{actual_hash}\n".encode())
    if digest.hexdigest() != manifest["payload_sha256"]:
        errors.append("payload composite SHA256 mismatch")
    for metadata_name in ("baseline_summary.json", "submission_receipt.json"):
        json_load(output_dir / metadata_name)
    if errors:
        raise ValueError("snapshot verification failed: " + "; ".join(errors))
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "payload_files": manifest["file_count"],
                "payload_sha256": manifest["payload_sha256"],
            },
            indent=2,
        )
    )


def freeze(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    run_dir = args.run_dir.resolve()
    config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing baseline snapshot: {output_dir}; use --verify"
        )

    checkpoint_path = run_dir / "checkpoints/best.pt"
    metrics_path = run_dir / "metrics/full_pq_epoch030.json"
    oof_manifest_path = run_dir / "oof_best/manifest.json"
    generation_audit_path = run_dir / "submission_modelA_fold0_generation_audit.json"
    reaudit_path = run_dir / "submission_modelA_fold0_reaudit.json"
    submission_path = run_dir / "submission_modelA_fold0_best.csv"

    checkpoint = checkpoint_metadata(checkpoint_path)
    metrics = json_load(metrics_path)
    oof_manifest = json_load(oof_manifest_path)
    generation_audit = json_load(generation_audit_path)
    reaudit = json_load(reaudit_path)
    run_manifest = json_load(run_dir / "run_manifest.json")
    validate_baseline(
        run_dir,
        checkpoint,
        metrics,
        oof_manifest,
        generation_audit,
        reaudit,
    )

    epoch_records = load_epoch_records(run_dir / "metrics/epochs.jsonl")
    full_reports = [
        json_load(path) for path in sorted((run_dir / "metrics").glob("full_pq_epoch*.json"))
    ]
    train_images = sorted((project_root / "MAGFiLO_1.0_Kaggle_2026/train/train_images").glob("*"))
    test_images = sorted((project_root / "MAGFiLO_1.0_Kaggle_2026/test/test_images").glob("*"))
    data_files = [
        project_root
        / "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        project_root / "data/manifests/physical_images.csv",
        project_root / "data/manifests/train_folds.csv",
        project_root / "data/manifests/train_folds.meta.json",
        project_root / "data/cache/medoid_v1/manifest.json",
    ]
    source_files = required_source_files(project_root, run_dir, config)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        payload_root = temporary / "payload"
        copy_payload(source_files, project_root, payload_root)

        frozen_checkpoint = payload_root / relative(checkpoint_path, project_root)
        frozen_metrics = payload_root / relative(metrics_path, project_root)
        frozen_oof_manifest = payload_root / relative(oof_manifest_path, project_root)
        frozen_submission = payload_root / relative(submission_path, project_root)
        frozen_config = payload_root / relative(run_dir / "config.resolved.yaml", project_root)
        frozen_generation_audit = payload_root / relative(generation_audit_path, project_root)
        frozen_reaudit = payload_root / relative(reaudit_path, project_root)

        def final_path(temporary_path: Path) -> Path:
            return output_dir / temporary_path.relative_to(temporary)

        def frozen_record(source: Path, temporary_path: Path) -> dict[str, Any]:
            return artifact_record(
                source,
                temporary_path,
                project_root,
                frozen_display_path=final_path(temporary_path),
            )

        data_fingerprint = {
            "metadata_files": {
                relative(path, project_root): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in data_files
            },
            "train_images": tree_fingerprint(train_images, project_root),
            "test_images": tree_fingerprint(test_images, project_root),
        }
        data_digest = hashlib.sha256()
        for path, record in sorted(data_fingerprint["metadata_files"].items()):
            data_digest.update(f"{path}\0{record['bytes']}\0{record['sha256']}\n".encode())
        data_digest.update(data_fingerprint["train_images"]["sha256"].encode())
        data_digest.update(data_fingerprint["test_images"]["sha256"].encode())
        data_fingerprint["combined_sha256"] = data_digest.hexdigest()

        archive = payload_manifest(payload_root, project_root)
        generated_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "schema_version": 1,
            "baseline_id": "model_a_fold0_epoch030",
            "frozen_at_utc": generated_at,
            "status": "frozen",
            "write_protection": "payload and metadata permissions set to read-only (0444/0555)",
            "competition": run_manifest["competition"],
            "model": {
                "name": "Model A",
                "fold": 0,
                "architecture": "resnet34_unet",
                "parameters": run_manifest["model"]["parameters"],
                "pretrained_weights": run_manifest["model"]["pretrained_weights"],
                "checkpoint_metadata": checkpoint,
            },
            "selection": {
                "criterion": "official_pq_v6 on complete fold-0 full-image OOF",
                "best_epoch": 30,
                "evaluated_epochs": [report["epoch"] for report in full_reports],
            },
            "oof_metrics": {
                "official_pq": metrics["official_pq"],
                "dice_annotator_macro": metrics["dice_annotator_macro"],
                "iou_annotator_macro": metrics["iou_annotator_macro"],
                "true_positives_hit_pairs": metrics["hit_pairs"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "hit_iou_sum": metrics["hit_iou_sum"],
                "physical_images": metrics["physical_images"],
                "annotator_records": metrics["annotator_records"],
                "fragmented_ground_truth_instances": metrics[
                    "one_to_many_gt_distribution"
                ]["multiple_partners"],
                "merged_prediction_instances": metrics[
                    "many_to_one_prediction_distribution"
                ]["multiple_partners"],
                "images_at_safety_cap": metrics["prediction_count_summary"][
                    "images_at_safety_cap"
                ],
                "metric_name": metrics["metric_name"],
                "iou_threshold": metrics["iou_threshold"],
                "matching": metrics["matching"],
            },
            "postprocess": metrics["inference_config"],
            "timings_seconds": {
                "logged_epoch_training_total_50_epochs": sum(
                    record["train_seconds"] for record in epoch_records
                ),
                "logged_epoch_training_through_best_epoch_30": sum(
                    record["train_seconds"]
                    for record in epoch_records
                    if record["epoch"] <= 30
                ),
                "full_image_validation_total_10_runs": sum(
                    report["evaluation_seconds"] for report in full_reports
                ),
                "best_epoch_full_image_validation": metrics["evaluation_seconds"],
                "test_inference_and_generation_audit": generation_audit["summary"][
                    "inference_and_audit_seconds"
                ],
                "standard_crop_validation": {
                    "value": None,
                    "reason": "not separately logged by the completed baseline run",
                },
            },
            "data_fingerprint": data_fingerprint,
            "code_version": git_state(project_root),
            "config": {
                "resolved": frozen_record(run_dir / "config.resolved.yaml", frozen_config),
                "canonical_fingerprint": run_manifest["config_fingerprint"],
            },
            "artifacts": {
                "checkpoint": frozen_record(checkpoint_path, frozen_checkpoint),
                "oof": {
                    "directory": relative(final_path(frozen_oof_manifest.parent), project_root),
                    "probability_files": metrics["physical_images"],
                    "probability_dtype": metrics["oof_probability_dtype"],
                    "manifest": frozen_record(oof_manifest_path, frozen_oof_manifest),
                    "payload_tree_sha256": tree_fingerprint(
                        frozen_oof_manifest.parent.glob("*"), frozen_oof_manifest.parent
                    )["sha256"],
                },
                "metric_report": frozen_record(metrics_path, frozen_metrics),
                "submission": frozen_record(submission_path, frozen_submission),
                "generation_audit": frozen_record(
                    generation_audit_path, frozen_generation_audit
                ),
                "independent_reaudit": frozen_record(reaudit_path, frozen_reaudit),
                "archive_manifest": "archive_manifest.json",
                "submission_receipt": "submission_receipt.json",
            },
            "archive_payload_sha256": archive["payload_sha256"],
        }

        receipt = {
            "schema_version": 1,
            "baseline_id": summary["baseline_id"],
            "recorded_at_utc": generated_at,
            "competition_slug": run_manifest["competition"]["slug"],
            "submission": frozen_record(submission_path, frozen_submission),
            "model_checkpoint": frozen_record(checkpoint_path, frozen_checkpoint),
            "resolved_config": frozen_record(
                run_dir / "config.resolved.yaml", frozen_config
            ),
            "oof": {
                "official_pq": metrics["official_pq"],
                "fold": 0,
                "best_epoch": 30,
                "manifest_sha256": sha256_file(frozen_oof_manifest),
            },
            "kaggle": {
                "status": "scored",
                "public_score": args.public_score,
                "score_source": args.score_source,
                "online_verified_during_freeze": False,
                "submission_id": None,
                "submitted_at": None,
                "verification_note": (
                    "Kaggle CLI was unavailable in the local environment; public score and "
                    "scored status were transcribed from the user-provided implementation plan."
                ),
            },
            "audit": {
                "generation": generation_audit["summary"],
                "independent_reaudit": reaudit,
                "passed": bool(reaudit["rles_validated"]),
            },
            "traceability": {
                "baseline_summary": "baseline_summary.json",
                "archive_manifest": "archive_manifest.json",
                "data_fingerprint_sha256": data_fingerprint["combined_sha256"],
                "archive_payload_sha256": archive["payload_sha256"],
            },
        }

        json_write(temporary / "archive_manifest.json", archive)
        json_write(temporary / "baseline_summary.json", summary)
        json_write(temporary / "submission_receipt.json", receipt)
        (temporary / "FROZEN").write_text(
            "Model A Fold 0 baseline selected at epoch 30. Do not overwrite.\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
        make_read_only(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    verify_snapshot(output_dir)


def main() -> None:
    args = parse_args()
    if args.verify:
        verify_snapshot(args.output_dir.resolve())
    else:
        freeze(args)


if __name__ == "__main__":
    main()

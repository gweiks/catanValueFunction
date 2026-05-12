"""flatten.py — Flatten per-game parquet files into a partitioned dataset.

Reads data/raw/<game_id>.parquet files, appends a shard column (sha256-based
% 100), writes a partitioned parquet dataset, generates splits.json (80/10/10
by game_id, reproducible), and writes MANIFEST.json with all DATA-18 fields
plus discretionary extras.

Public API:
    SPLIT_SEED              int  — fixed seed for deterministic splits
    DATASET_VERSION         str  — semver; embedded in MANIFEST
    FEATURE_SUBSET_NAME     str  — matches data/feature_ordering.json
    shard_of(game_id)       str  → int in range(100); sha256-based, never hash()
    build_splits(game_ids)  list[str] → dict[str,list[str]] — 80/10/10 by game_id
    assert_no_overlap(splits) — raises AssertionError if any id is in two splits
    write_manifest(...)     — writes MANIFEST.json atomically
    flatten_run(...)        — orchestrates the full flatten pipeline; returns summary
"""

import hashlib
import json
import os
import random as _rnd
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from schema import SCHEMA

# Module-level constants

SPLIT_SEED: int = 0xCA7A_5EED
"""Fixed seed for train/val/test split shuffling. NEVER change this after first
dataset creation — changing it would produce different splits and invalidate the
anti-p-hacking guard (test split must be committed before it is inspected)."""

DATASET_VERSION: str = "1.0.0"
"""Semver dataset version. Increment major when schema or split logic changes."""

FEATURE_SUBSET_NAME: str = "subset_92"
"""Name of the 92-feature subset used; matches data/feature_ordering.json."""


# ---------------------------------------------------------------------------
# shard_of — sha256-based, never Python builtin hash()
# ---------------------------------------------------------------------------

# Return the shard index (0-99) for a given game_id.
def shard_of(game_id: str) -> int:
    digest = hashlib.sha256(game_id.encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100



# Partition game_ids into train/val/test lists (80/10/10).
def build_splits(game_ids: list[str]) -> dict[str, list[str]]:
    sorted_ids = sorted(game_ids)
    rng = _rnd.Random(SPLIT_SEED)
    rng.shuffle(sorted_ids)

    n = len(sorted_ids)
    n_val = max(1, round(n * 0.10)) if n >= 2 else 0
    n_test = max(1, round(n * 0.10)) if n >= 2 else 0

    # For exactly 100 games we want 80/10/10 precisely.
    # General formula: train gets everything not in val+test.
    if n == 100:
        n_val = 10
        n_test = 10

    train = sorted_ids[: n - n_val - n_test]
    val = sorted_ids[n - n_val - n_test : n - n_test]
    test = sorted_ids[n - n_test :]

    return {"train": train, "val": val, "test": test}



def assert_no_overlap(splits: dict[str, list[str]]) -> None:
    """Raise AssertionError if any game_id appears in more than one split."""
    seen: set[str] = set()
    for split_name, ids in splits.items():
        overlap = seen & set(ids)
        assert not overlap, (
            f"game_ids appear in multiple splits: {overlap} (found in '{split_name}')"
        )
        seen |= set(ids)



def write_manifest(
    manifest_path: Path,
    totals: dict,
    splits_path: Path,
    feature_ordering_path: Path,
):
    """
    Fields written:
        dataset_version, git_commit, total_games,
        total_snapshots, feature_subset_name, splits_sha256,
        feature_ordering_sha256, generation_timestamp
    Discretionary extras: truncation_rate, per_pairing_snapshot_counts,
            n_failed_games, schema_sha256

    """
    # sha256 of splits.json
    splits_sha256 = hashlib.sha256(splits_path.read_bytes()).hexdigest()

    # sha256 of feature_ordering.json (use bytes for stability)
    if feature_ordering_path.exists():
        feature_ordering_sha256 = hashlib.sha256(
            feature_ordering_path.read_bytes()
        ).hexdigest()
    else:
        feature_ordering_sha256 = "UNKNOWN"

    # sha256 of schema (field name:type concatenation — stable across pyarrow versions)
    schema_sha256 = hashlib.sha256(
        "|".join(f"{f.name}:{f.type}" for f in SCHEMA).encode()
    ).hexdigest()

    # git commit
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(manifest_path.parent.parent if manifest_path.parent != manifest_path.parent.parent else "."),
        ).stdout.strip() or "UNKNOWN"
    except Exception:
        git_commit = "UNKNOWN"

    manifest = {
        "dataset_version": DATASET_VERSION,
        "feature_ordering_sha256": feature_ordering_sha256,
        "feature_subset_name": FEATURE_SUBSET_NAME,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "n_failed_games": totals.get("n_failed_games", 0),
        "per_pairing_snapshot_counts": totals.get("per_pairing_snapshot_counts", {}),
        "schema_sha256": schema_sha256,
        "splits_sha256": splits_sha256,
        "total_games": totals["total_games"],
        "total_snapshots": totals["total_snapshots"],
        "truncation_rate": totals.get("truncation_rate", 0.0),
    }

    content = json.dumps(manifest, indent=2, sort_keys=True)
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, manifest_path)



# Flatten per-game parquet files into a partitioned dataset.les
# AssertionError if integrity checks fail.
def flatten_run(
    raw_dir: Path,
    out_dir: Path,
    data_dir: Path | None = None,
) -> dict:
    """Flatten per-game parquet files into a partitioned dataset.les.
        AssertionError: if integrity checks fail.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    if data_dir is None:
        data_dir = out_dir.parent

    data_dir = Path(data_dir)

    # Read raw dataset (validates against SCHEMA on read)
    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"no parquet files under {raw_dir}")

    raw_dataset = ds.dataset(str(raw_dir), schema=SCHEMA, format="parquet")
    table = raw_dataset.to_table()

    # Append shard column
    game_ids_col = table.column("game_id").to_pylist()
    shards = pa.array([shard_of(g) for g in game_ids_col], type=pa.uint8())
    table = table.append_column(
        pa.field("shard", pa.uint8(), nullable=False), shards
    )

    # 3. Write partitioned parquet
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_to_dataset(
        table,
        root_path=str(out_dir),
        partition_cols=["shard"],
        existing_data_behavior="delete_matching",
        compression="snappy",
    )

    # Build splits and write splits.json
    unique_ids = sorted(set(game_ids_col))
    splits = build_splits(unique_ids)
    assert_no_overlap(splits)

    splits_path = data_dir / "splits.json"
    splits_path.write_text(
        json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Compute totals and extras
    total_games = len(unique_ids)
    total_snapshots = table.num_rows

    # truncation_rate: fraction of games that did NOT complete within TURNS_LIMIT
    game_complete_col = table.column("game_complete").to_pylist()
    game_complete_map: dict[str, bool] = {}
    for gid, gc in zip(game_ids_col, game_complete_col):
        # If any snapshot in the game has game_complete=True, the game completed.
        game_complete_map[gid] = game_complete_map.get(gid, False) or bool(gc)

    n_complete_games = sum(1 for v in game_complete_map.values() if v)
    truncation_rate = (
        (total_games - n_complete_games) / total_games if total_games else 0.0
    )

    # per_pairing_snapshot_counts: dict["{p1}|{p2}", int]
    p1_col = table.column("p1_bot").to_pylist()
    p2_col = table.column("p2_bot").to_pylist()
    per_pairing: dict[str, int] = {}
    for p1, p2 in zip(p1_col, p2_col):
        key = f"{p1}|{p2}"
        per_pairing[key] = per_pairing.get(key, 0) + 1

    n_failed_games = 0  # collect.py excludes failed games (no parquet written)

    # Integrity assertions
    tmp_survivors = list(raw_dir.glob("*.tmp"))
    assert not tmp_survivors, f".tmp survivors in {raw_dir}: {tmp_survivors}"

    # Partitioned dataset schema equals SCHEMA (minus shard if present)
    out_schema = ds.dataset(str(out_dir), format="parquet").schema
    shard_idx = out_schema.get_field_index("shard")
    if shard_idx >= 0:
        base_schema = out_schema.remove(shard_idx)
    else:
        base_schema = out_schema
    assert base_schema.equals(SCHEMA, check_metadata=False), (
        f"partitioned dataset schema drifted from SCHEMA.\n"
        f"Got:      {base_schema}\n"
        f"Expected: {SCHEMA}"
    )

    # No duplicates within a split; no overlap across splits
    seen: set[str] = set()
    for split_name, ids in splits.items():
        assert len(ids) == len(set(ids)), f"Duplicate game_ids within split '{split_name}'"
        overlap = seen & set(ids)
        assert not overlap, f"game_ids overlap between splits: {overlap}"
        seen |= set(ids)

    # Splits cover exactly the full unique_id set
    assert seen == set(unique_ids), (
        f"splits do not cover all game_ids.\n"
        f"Missing: {set(unique_ids) - seen}\n"
        f"Extra:   {seen - set(unique_ids)}"
    )

    # Write MANIFEST.json
    manifest_path = data_dir / "MANIFEST.json"
    feature_ordering_path = Path("data/feature_ordering.json")

    write_manifest(
        manifest_path=manifest_path,
        totals={
            "total_games": total_games,
            "total_snapshots": total_snapshots,
            "truncation_rate": truncation_rate,
            "per_pairing_snapshot_counts": per_pairing,
            "n_failed_games": n_failed_games,
        },
        splits_path=splits_path,
        feature_ordering_path=feature_ordering_path,
    )

    return {
        "total_games": total_games,
        "total_snapshots": total_snapshots,
        "truncation_rate": truncation_rate,
        "n_failed_games": n_failed_games,
        "manifest_path": str(manifest_path),
        "splits_path": str(splits_path),
    }



# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flatten raw per-game parquet files.")
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw"),
        help="Directory of per-game parquet files (default: data/raw)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/snapshots.parquet"),
        help="Root path for the partitioned parquet dataset"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Where to write splits.json + MANIFEST.json (default: out_dir.parent)"
    )
    args = parser.parse_args()
    result = flatten_run(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
    )
    print(json.dumps(result, indent=2))

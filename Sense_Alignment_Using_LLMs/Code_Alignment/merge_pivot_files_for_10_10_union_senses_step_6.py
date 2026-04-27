from pathlib import Path
import pandas as pd

# CONFIG
BASE_DIR = Path(r"PATH\Alignment_Statistics_Results\csv_files")

PIVOT_FOLDER_PREFIX = "strict_union_with_pivot_"
STRICT_FILENAME = "verified_global_5of5.csv"

# Pivot preference when duplicates exist 
PIVOT_PRIORITY = ["bg", "el", "it", "fr", "ar"]

OUT_MASTER = BASE_DIR / "MASTER_strict_10of10_all_pivots_dedup.csv"
OUT_SUMMARY = BASE_DIR / "MASTER_strict_10of10_summary_by_pivot.csv"

# HELPERS
def infer_pivot_from_folder(folder: Path) -> str:
    # strict_union_with_pivot_el -> el
    name = folder.name
    if name.startswith(PIVOT_FOLDER_PREFIX):
        return name[len(PIVOT_FOLDER_PREFIX):]
    return "unknown"

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the strict 5-tuple row_id columns (+ attempt_id if present).
    """
    wanted = ["ar_row_id", "bg_row_id", "el_row_id", "fr_row_id", "it_row_id"]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns {missing}. Columns seen: {df.columns.tolist()}")

    keep = []
    if "attempt_id" in df.columns:
        keep.append("attempt_id")
    keep += wanted

    return df[keep].copy()

def make_tuple_key(df: pd.DataFrame) -> pd.Series:
    # canonical key: ar=..|bg=..|el=..|fr=..|it=..
    return (
        "ar=" + df["ar_row_id"].astype(str) + "|" +
        "bg=" + df["bg_row_id"].astype(str) + "|" +
        "el=" + df["el_row_id"].astype(str) + "|" +
        "fr=" + df["fr_row_id"].astype(str) + "|" +
        "it=" + df["it_row_id"].astype(str)
    )

def pivot_rank(p: str) -> int:
    try:
        return PIVOT_PRIORITY.index(p)
    except ValueError:
        return 999

# MAIN
def run():
    pivot_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith(PIVOT_FOLDER_PREFIX)])
    if not pivot_dirs:
        raise FileNotFoundError(f"No pivot folders found under: {BASE_DIR}")

    frames = []
    summary_rows = []

    for d in pivot_dirs:
        pivot = infer_pivot_from_folder(d)
        strict_path = d / STRICT_FILENAME
        if not strict_path.exists():
            print(f"[SKIP] No {STRICT_FILENAME} in {d}")
            continue

        df = pd.read_csv(strict_path, dtype=str)  # dtype=str keeps IDs stable
        df = normalize_cols(df)
        df["pivot"] = pivot
        df["status"] = "strict_10of10"

        # build key and attach
        df["tuple_key"] = make_tuple_key(df)

        frames.append(df)

        summary_rows.append({
            "pivot": pivot,
            "rows_strict": len(df),
            "file": str(strict_path)
        })

        print(f"[OK] {pivot}: loaded {len(df)} strict rows from {strict_path}")

    if not frames:
        raise RuntimeError("No strict files loaded. Check folder names and paths.")

    all_df = pd.concat(frames, ignore_index=True)

    # Sort so that for duplicates we keep the best pivot based on PIVOT_PRIORITY
    all_df["pivot_rank"] = all_df["pivot"].map(pivot_rank)
    all_df = all_df.sort_values(["pivot_rank"], ascending=True)

    # Deduplicate by tuple_key
    before = len(all_df)
    dedup_df = all_df.drop_duplicates(subset=["tuple_key"], keep="first").copy()
    after = len(dedup_df)

    # Cleanup
    dedup_df = dedup_df.drop(columns=["pivot_rank"])

    # Save master + summary
    dedup_df.to_csv(OUT_MASTER, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).sort_values("rows_strict", ascending=False).to_csv(
        OUT_SUMMARY, index=False, encoding="utf-8-sig"
    )

    print("\n=== DONE ===")
    print(f"Total strict rows (all pivots, before dedup): {before}")
    print(f"Unique strict 5-tuples (after dedup):         {after}")
    print(f"Saved master:  {OUT_MASTER}")
    print(f"Saved summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    run()

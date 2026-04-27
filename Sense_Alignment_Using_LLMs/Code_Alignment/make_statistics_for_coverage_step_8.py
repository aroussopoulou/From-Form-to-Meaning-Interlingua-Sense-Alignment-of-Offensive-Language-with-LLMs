from pathlib import Path
import pandas as pd

LANGS = ["ar", "bg", "el", "fr", "it"]
LANGNAME = {"ar":"Arabic","bg":"Bulgarian","el":"Greek","fr":"French","it":"Italian"}

# Config
UNION_CSV = Path(r"D:\Documents\AssignmentsForMaster\Thesis\Alignment_Statistics_Results\stats_strict_kof5_ALL_PIVOTS\UNION_ALL_PIVOTS_strict_3_4_5of5_DEDUP.csv")
DATASET_CSV = Path(r"C:\Users\changeme\Documents\AssignmentsForMaster\Thesis\Alignment_File\hurtnet_total_dataset.csv")
OUT_DIR = Path(r"D:\Documents\AssignmentsForMaster\Thesis\Alignment_Statistics_Results\stats_strict_kof5_ALL_PIVOTS")

OUT_COVERAGE = OUT_DIR / "COVERAGE_by_language_and_k.csv"
OUT_MISSING  = OUT_DIR / "MISSING_language_patterns.csv"

def load_dataset_counts(dataset_csv: Path):
    ds = pd.read_csv(dataset_csv)
    cols = {c.lower(): c for c in ds.columns}
    lang_col = cols.get("language")
    row_col  = cols.get("row_id") or cols.get("rowid") or cols.get("row id")
    if not (lang_col and row_col):
        raise ValueError(f"Dataset must contain Language + row_id. Columns: {list(ds.columns)}")

    ds = ds[[lang_col, row_col]].copy()
    ds.columns = ["Language", "Row_id"]
    ds["Row_id"] = ds["Row_id"].astype(str)

    # map Language name -> code
    name2code = {v:k for k,v in LANGNAME.items()}

    counts = {}
    for code in LANGS:
        langname = LANGNAME[code]
        sub = ds[ds["Language"] == langname]
        counts[code] = {
            "language": langname,
            "total_row_ids_in_dataset": sub["Row_id"].nunique()
        }
    return counts

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(UNION_CSV, dtype=str)
    # expected columns
    for l in LANGS:
        if f"{l}_row_id" not in df.columns:
            raise ValueError(f"Missing column {l}_row_id in union file")

    # Normalize blanks
    for l in LANGS:
        col = f"{l}_row_id"
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Coverage table
    dataset_counts = load_dataset_counts(DATASET_CSV)

    rows = []
    for k in ["5/5","4/5","3/5"]:
        dfk = df[df["k_of_5"] == k].copy()
        for l in LANGS:
            ids = dfk[f"{l}_row_id"]
            ids = ids[(ids != "") & (ids != "nan")]
            uniq = ids.nunique()

            total = dataset_counts[l]["total_row_ids_in_dataset"]
            pct = (uniq / total * 100.0) if total else 0.0

            rows.append({
                "k_of_5": k,
                "lang": l,
                "language": dataset_counts[l]["language"],
                "unique_row_ids_covered": int(uniq),
                "total_row_ids_in_dataset": int(total),
                "coverage_pct": round(pct, 4),
                "n_tuples_in_k": int(len(dfk)),
            })

    cov = pd.DataFrame(rows).sort_values(["k_of_5","lang"])
    cov.to_csv(OUT_COVERAGE, index=False, encoding="utf-8-sig")

    # Missing language patterns
    # 4/5 => single missing lang
    miss_rows = []
    df4 = df[df["k_of_5"] == "4/5"].copy()
    if len(df4):
        m = df4["langs_missing"].fillna("").astype(str).str.strip()
        m = m[m != ""]
        counts = m.value_counts()
        for miss, n in counts.items():
            miss_rows.append({"k_of_5":"4/5", "missing": miss, "count": int(n)})

    # 3/5 => two missing langs
    df3 = df[df["k_of_5"] == "3/5"].copy()
    if len(df3):
        m = df3["langs_missing"].fillna("").astype(str).str.strip()
        m = m[m != ""]
        counts = m.value_counts()
        for miss, n in counts.items():
            miss_rows.append({"k_of_5":"3/5", "missing": miss, "count": int(n)})

    missdf = pd.DataFrame(miss_rows).sort_values(["k_of_5","count"], ascending=[True, False])
    missdf.to_csv(OUT_MISSING, index=False, encoding="utf-8-sig")

    print("[OK] Saved:", OUT_COVERAGE)
    print("[OK] Saved:", OUT_MISSING)
    print("Done.")

if __name__ == "__main__":
    main()
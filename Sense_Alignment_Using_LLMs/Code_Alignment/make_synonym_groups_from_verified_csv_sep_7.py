import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse
import re

LANGS = ["ar", "bg", "el", "fr", "it"]

# Union-Find / DSU

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

# Helpers

def normalize_row_id(x) -> str:
    """
    Normalizes row IDs so that 12, '12', and '12.0' can match.
    """
    if pd.isna(x):
        return ""

    x = str(x).strip()

    if x.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""

    # Convert strings like "12.0" to "12"
    if re.fullmatch(r"\d+\.0", x):
        x = x[:-2]

    return x


def list_fmt(items):
    """
    Stable unique list formatting for CSV cells.
    Example: [34, 55, 89]
    """
    seen = set()
    out = []

    for x in items:
        x = "" if x is None else str(x).strip()

        if not x:
            continue

        if x not in seen:
            seen.add(x)
            out.append(x)

    return "[" + ", ".join(out) + "]"


def normalize_language_code(language_value: str) -> str:
    """
    Maps language names or codes to the fixed language codes:
    ar, bg, el, fr, it.
    """
    x = str(language_value).strip().lower()

    mapping = {
        "arabic": "ar",
        "ar": "ar",

        "bulgarian": "bg",
        "bg": "bg",

        "greek": "el",
        "modern greek": "el",
        "el": "el",

        "french": "fr",
        "fr": "fr",

        "italian": "it",
        "it": "it",
    }

    return mapping.get(x, "")


def build_rowid2lemma(dataset_df: pd.DataFrame):
    """
    Builds a dictionary:
        (language_code, row_id) -> lemma

    Expected dataset columns:
        Language, Row_id, Lemma

    It also accepts small variations:
        row_id / rowid / row id
        lemma / term / word
    """
    cols = {c.lower().strip(): c for c in dataset_df.columns}

    lang_col = cols.get("language")
    row_col = cols.get("row_id") or cols.get("rowid") or cols.get("row id")
    lemma_col = cols.get("lemma") or cols.get("term") or cols.get("word")

    if not (lang_col and row_col and lemma_col):
        raise ValueError(
            "Dataset must contain Language, Row_id, and Lemma columns "
            "or equivalent names. "
            f"Found columns: {list(dataset_df.columns)}"
        )

    ds = dataset_df[[lang_col, row_col, lemma_col]].copy()
    ds.columns = ["Language", "Row_id", "Lemma"]

    ds["Language"] = ds["Language"].apply(normalize_language_code)
    ds["Row_id"] = ds["Row_id"].apply(normalize_row_id)
    ds["Lemma"] = ds["Lemma"].astype(str).str.strip()

    rowid2lemma = {}

    for r in ds.itertuples(index=False):
        if not r.Language or not r.Row_id:
            continue

        rowid2lemma[(r.Language, r.Row_id)] = r.Lemma if r.Lemma else ""

    return rowid2lemma


def load_strict_tuples(master_csv: Path) -> pd.DataFrame:
    """
    Loads the strict 10/10 tuple file.

    Expected columns:
        ar_row_id, bg_row_id, el_row_id, fr_row_id, it_row_id
    """
    df = pd.read_csv(master_csv, dtype=str)

    needed = [f"{lang}_row_id" for lang in LANGS]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {master_csv}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[needed].copy()

    for c in needed:
        df[c] = df[c].apply(normalize_row_id)

    # Remove rows with empty/bad IDs
    mask = df[needed].apply(lambda col: col.astype(str).str.strip() != "").all(axis=1)
    df = df.loc[mask].copy()

    # Remove exact duplicate strict tuples
    df = df.drop_duplicates().reset_index(drop=True)

    return df

# Core 4/5 GlobalSense clustering

def cluster_global_senses_4of5(df: pd.DataFrame) -> UnionFind:
    """
    Groups strict 5-language tuples into GlobalSense clusters.

    Rule:
    Two strict tuples are linked if they share the same row_id values
    in any four out of the five languages.

    Example:
        T  = (ar:34, bg:930, el:1323, fr:2222, it:2674)
        T' = (ar:34, bg:930, el:1323, fr:2222, it:2783)

    They share ar, bg, el, fr and differ only in it.
    Therefore, they belong to the same GlobalSense.

    The Union-Find structure applies this transitively:
        if T links to T' and T' links to T'', all three are placed
        in the same GlobalSense cluster.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    uf = UnionFind(n)

    # One signature map per omitted language.
    # If we omit "it", the signature is based on ar, bg, el, fr.
    signature_maps = {omitted_lang: {} for omitted_lang in LANGS}

    for i, row in df.iterrows():
        tuple_values = {
            "ar": row["ar_row_id"],
            "bg": row["bg_row_id"],
            "el": row["el_row_id"],
            "fr": row["fr_row_id"],
            "it": row["it_row_id"],
        }

        for omitted_lang in LANGS:
            signature = tuple(
                tuple_values[lang]
                for lang in LANGS
                if lang != omitted_lang
            )

            seen_map = signature_maps[omitted_lang]

            if signature in seen_map:
                uf.union(i, seen_map[signature])
            else:
                seen_map[signature] = i

    return uf


# Build GlobalSense table

def build_globalsense_table(
    df: pd.DataFrame,
    uf: UnionFind,
    rowid2lemma: dict,
    min_tuples: int = 1
) -> pd.DataFrame:
    """
    Creates a human-readable table of GlobalSense clusters.

    Each row corresponds to one GlobalSense.
    For every language, it stores:
        - row IDs included in the cluster
        - lemmas included in the cluster
        - count of unique row IDs
    """
    comps = defaultdict(list)

    for i in range(len(df)):
        root = uf.find(i)
        comps[root].append(i)

    rows = []
    gs_id = 0

    for root, idxs in comps.items():
        if len(idxs) < min_tuples:
            continue

        gs_id += 1

        row_out = {
            "globalsense_id": gs_id,
            "n_tuples": len(idxs),
        }

        # Collect row IDs and lemmas per language
        for lang in LANGS:
            ids = []
            seen = set()
            col = f"{lang}_row_id"

            for i in idxs:
                rid = normalize_row_id(df.at[i, col])

                if rid and rid not in seen:
                    seen.add(rid)
                    ids.append(rid)

            lemmas = [
                rowid2lemma.get((lang, rid), "")
                for rid in ids
            ]
            lemmas = [x for x in lemmas if str(x).strip()]

            row_out[f"{lang}_row_ids"] = list_fmt(ids)
            row_out[f"{lang}_lemmas"] = list_fmt(lemmas)
            row_out[f"{lang}_count"] = len(ids)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["n_tuples", "el_count", "fr_count", "it_count", "bg_count", "ar_count"],
            ascending=False
        ).reset_index(drop=True)

        # Reassign IDs after sorting, so GlobalSense 1 is the largest cluster
        out["globalsense_id"] = range(1, len(out) + 1)

    return out


# =========================
# Main
# =========================

def main(master_csv: Path, dataset_csv: Path, out_csv: Path, min_tuples: int):
    print("[1/4] Loading strict 10/10 tuples:")
    print("     ", master_csv)

    df = load_strict_tuples(master_csv)
    print("     strict unique tuples:", len(df))

    print("\n[2/4] Loading dataset and building row_id -> lemma map:")
    print("     ", dataset_csv)

    ds = pd.read_csv(dataset_csv, dtype=str)
    rowid2lemma = build_rowid2lemma(ds)
    print("     lemma map size:", len(rowid2lemma))

    print("\n[3/4] Clustering strict tuples with the 4/5 GlobalSense rule...")
    uf = cluster_global_senses_4of5(df)

    print("\n[4/4] Building GlobalSense table...")
    outdf = build_globalsense_table(
        df=df,
        uf=uf,
        rowid2lemma=rowid2lemma,
        min_tuples=min_tuples
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n=== DONE ===")
    print("GlobalSenses:", len(outdf))
    print("Saved:", out_csv)

    if not outdf.empty:
        print("\nTop 10 GlobalSenses by size:")
        cols = [
            "globalsense_id",
            "n_tuples",
            "el_rep_row_id",
            "el_rep_lemma",
            "ar_count",
            "bg_count",
            "el_count",
            "fr_count",
            "it_count",
        ]

        existing_cols = [c for c in cols if c in outdf.columns]
        print(outdf[existing_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--master",
        default=r"D:\Documents\AssignmentsForMaster\Thesis\Alignment_Statistics_Results\csv_files\MASTER_strict_10of10_all_pivots_dedup.csv",
        help="CSV file with strict 10/10 deduplicated tuples."
    )

    ap.add_argument(
        "--dataset",
        default=r"C:\Users\changeme\Documents\AssignmentsForMaster\Thesis\Alignment_File\hurtnet_total_dataset.csv",
        help="Hurtnet dataset used to map row_id -> lemma."
    )

    ap.add_argument(
        "--out",
        default=r"D:\Documents\AssignmentsForMaster\Thesis\Alignment_Statistics_Results\csv_files\GLOBAL_SENSES_strict_4of5_1.csv",
        help="Output CSV file for GlobalSense clusters."
    )

    ap.add_argument(
        "--min_tuples",
        type=int,
        default=1,
        help=(
            "Minimum number of strict tuples required to keep a GlobalSense. "
            "Use 1 to keep all clusters. Use 2 to remove singleton clusters."
        )
    )

    args = ap.parse_args()

    main(
        master_csv=Path(args.master),
        dataset_csv=Path(args.dataset),
        out_csv=Path(args.out),
        min_tuples=args.min_tuples
    )
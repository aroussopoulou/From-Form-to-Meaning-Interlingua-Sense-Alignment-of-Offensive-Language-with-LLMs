import json
from pathlib import Path
import pandas as pd

JSONL_DIR   = Path(r"PATH\Alignment_Statistics_Results\jsonl_files")
DATASET_CSV = Path(r"PATH\Alignment_File\hurtnet_total_dataset.csv")

OUT_DIR = Path(r"PATHAlignment_Statistics_Results\stats_never_in_merge_version_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["ar", "bg", "el", "fr", "it"]
LANGNAME = {"ar":"Arabic", "bg":"Bulgarian", "el":"Greek", "fr":"French", "it":"Italian"}

EN_COL_CANDIDATES = ["Lemma_en"]

DEF_EN_COL_CANDIDATES = ["Definition_en"]

DEF_NATIVE_COL_CANDIDATES = ["Definition"]

def pair_from_alignment_id(alignment_id: str):
    prefix = alignment_id.split("_")[0]
    return prefix[:2], prefix[2:4]

def pick_column(ds: pd.DataFrame, candidates):
    cols_lower = {c.lower(): c for c in ds.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def load_dataset_maps(dataset_csv: Path):
    ds = pd.read_csv(dataset_csv)

    cols = {c.lower(): c for c in ds.columns}
    lang_col  = cols.get("language")
    row_col   = cols.get("row_id") or cols.get("rowid") or cols.get("row id")
    lemma_col = cols.get("lemma") or cols.get("term") or cols.get("word")

    if not (lang_col and row_col and lemma_col):
        raise ValueError(f"Dataset must contain Language/Row_id/Lemma. Found: {list(ds.columns)}")

    en_col       = pick_column(ds, EN_COL_CANDIDATES)
    def_en_col   = pick_column(ds, DEF_EN_COL_CANDIDATES)
    def_nat_col  = pick_column(ds, DEF_NATIVE_COL_CANDIDATES)

    # keep only what exists
    base_cols = [lang_col, row_col, lemma_col]
    if en_col:      base_cols.append(en_col)
    if def_en_col:  base_cols.append(def_en_col)
    if def_nat_col: base_cols.append(def_nat_col)

    ds = ds[base_cols].copy()

    # rename to normalized internal names
    rename_map = {lang_col: "Language", row_col: "Row_id", lemma_col: "Lemma"}
    if en_col:      rename_map[en_col] = "Lemma_en"
    if def_en_col:  rename_map[def_en_col] = "Definition_en"
    if def_nat_col: rename_map[def_nat_col] = "Definition"

    ds = ds.rename(columns=rename_map)
    ds["Row_id"] = ds["Row_id"].astype(str)

    name2code = {v: k for k, v in LANGNAME.items()}

    all_rowids = {l: set() for l in LANGS}
    rowid2lemma = {l: {} for l in LANGS}
    rowid2lemma_en = {l: {} for l in LANGS}
    rowid2def_en = {l: {} for l in LANGS}
    rowid2def_native = {l: {} for l in LANGS}

    # helper to safely stringify
    def s(x):
        if x is None:
            return ""
        if isinstance(x, float) and pd.isna(x):
            return ""
        if pd.isna(x):
            return ""
        return str(x)

    for r in ds.itertuples(index=False):
        code = name2code.get(str(r.Language))
        if code not in LANGS:
            continue

        rid = str(r.Row_id)
        all_rowids[code].add(rid)

        rowid2lemma[code][rid] = s(getattr(r, "Lemma", ""))

        if en_col:
            rowid2lemma_en[code][rid] = s(getattr(r, "Lemma_en", ""))
        else:
            rowid2lemma_en[code][rid] = ""

        if def_en_col:
            rowid2def_en[code][rid] = s(getattr(r, "Definition_en", ""))
        else:
            rowid2def_en[code][rid] = ""

        if def_nat_col:
            rowid2def_native[code][rid] = s(getattr(r, "Definition", ""))
        else:
            rowid2def_native[code][rid] = ""

    return all_rowids, rowid2lemma, rowid2lemma_en, rowid2def_en, rowid2def_native, (en_col, def_en_col, def_nat_col)

def scan_jsonls_for_participation(jsonl_dir: Path):
    merge_participants = {l: set() for l in LANGS}
    related_participants = {l: set() for l in LANGS}
    unrelated_participants = {l: set() for l in LANGS}
    any_participants = {l: set() for l in LANGS}

    total_lines = 0
    kept_lines = 0
    decision_counts = {"merge": 0, "related": 0, "unrelated": 0}

    paths = sorted(jsonl_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No .jsonl files found in: {jsonl_dir}")

    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue

                resp = d.get("response", {}) or {}
                decision = (resp.get("decision") or "").strip().lower()
                if decision not in decision_counts:
                    continue

                alignment_id = d.get("alignment_id")
                if not alignment_id:
                    continue

                la, lb = pair_from_alignment_id(alignment_id)
                if la not in LANGS or lb not in LANGS:
                    continue

                ra = d.get(f"{la}_row_id")
                rb = d.get(f"{lb}_row_id")
                if ra is None or rb is None:
                    continue

                ra = str(ra); rb = str(rb)

                kept_lines += 1
                decision_counts[decision] += 1

                any_participants[la].add(ra); any_participants[lb].add(rb)

                if decision == "merge":
                    merge_participants[la].add(ra); merge_participants[lb].add(rb)
                elif decision == "related":
                    related_participants[la].add(ra); related_participants[lb].add(rb)
                else:
                    unrelated_participants[la].add(ra); unrelated_participants[lb].add(rb)

    debug = {
        "total_lines_scanned": total_lines,
        "decision_lines_kept": kept_lines,
        "decision_counts": decision_counts,
        "files": [str(x) for x in paths]
    }
    return merge_participants, related_participants, unrelated_participants, any_participants, debug


def main():
    print("Loading dataset...")
    all_rowids, rowid2lemma, rowid2lemma_en, rowid2def_en, rowid2def_native, detected = load_dataset_maps(DATASET_CSV)
    en_col, def_en_col, def_nat_col = detected
    print(f"[INFO] English lemma column detected: {en_col}" if en_col else "[INFO] No English lemma column detected.")
    print(f"[INFO] English definition column detected: {def_en_col}" if def_en_col else "[INFO] No English definition column detected.")
    print(f"[INFO] Native definition column detected: {def_nat_col}" if def_nat_col else "[INFO] No native definition column detected.")

    print("Scanning JSONLs...")
    merge_part, rel_part, unrel_part, any_part, debug = scan_jsonls_for_participation(JSONL_DIR)
    print("[INFO] decision counts:", debug["decision_counts"])

    summary_rows = []
    for l in LANGS:
        total = len(all_rowids[l])
        in_merge = len(merge_part[l])
        never_merge = len(all_rowids[l] - merge_part[l])

        summary_rows.append({
            "lang": l,
            "total_row_ids_in_dataset": total,
            "row_ids_appeared_in_merge": in_merge,
            "row_ids_never_in_merge": never_merge,
            "pct_never_in_merge": (never_merge / total * 100.0) if total else 0.0,
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("lang")
    out_summary = OUT_DIR / "SUMMARY_no_merge_by_language.csv"
    df_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    # ---------- LIST: NEVER IN MERGE ----------
    rows_never_merge = []
    rows_never_merge_but_related = []
    rows_never_merge_but_unrelated = []

    for l in LANGS:
        never_ids = sorted(all_rowids[l] - merge_part[l])
        for rid in never_ids:
            lemma = rowid2lemma[l].get(rid, "")
            lemma_en = rowid2lemma_en[l].get(rid, "")
            def_en = rowid2def_en[l].get(rid, "")
            def_native = rowid2def_native[l].get(rid, "")

            base_row = {
                "lang": l,
                "row_id": rid,
                "lemma": lemma,
                "lemma_en": lemma_en,
                "definition_en": def_en,
                "definition_native": def_native
            }

            rows_never_merge.append(base_row)

            if rid in rel_part[l]:
                rows_never_merge_but_related.append(dict(base_row))
            if rid in unrel_part[l]:
                rows_never_merge_but_unrelated.append(dict(base_row))

    out_never = OUT_DIR / "LEMMA_LIST_never_in_merge.csv"
    pd.DataFrame(rows_never_merge).to_csv(out_never, index=False, encoding="utf-8-sig")

    out_never_rel = OUT_DIR / "LEMMA_LIST_never_in_merge_but_in_related.csv"
    pd.DataFrame(rows_never_merge_but_related).to_csv(out_never_rel, index=False, encoding="utf-8-sig")

    out_never_unrel = OUT_DIR / "LEMMA_LIST_never_in_merge_but_in_unrelated.csv"
    pd.DataFrame(rows_never_merge_but_unrelated).to_csv(out_never_unrel, index=False, encoding="utf-8-sig")

    # ---------- DEBUG ----------
    dbg = OUT_DIR / "DEBUG_scan_info.txt"
    dbg.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print("Saved:", out_summary)
    print("Saved:", out_never)
    print("Saved:", out_never_rel)
    print("Saved:", out_never_unrel)
    print("Saved:", dbg)

if __name__ == "__main__":
    main()
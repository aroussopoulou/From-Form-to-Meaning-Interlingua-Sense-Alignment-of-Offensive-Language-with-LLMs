from __future__ import annotations
from pathlib import Path
import pandas as pd


# =========================
# CONFIG (your uploaded files)
# =========================
HURTNET_CSV = Path(r"PATH\Alignment_File\hurtnet_total_dataset.csv")
TSV_FILE    = Path(r"PATH\translations (1).tsv")

OUT_DIR = Path("PATH\Alignment_Statistics_Results\csv_files\Comparison_per_HurtLex_and_HurtNet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL5_CSV = OUT_DIR / "tsv_global_merges_all5_in_hurtnet.csv"
OUT_DROPPED_CSV = OUT_DIR / "tsv_rows_dropped_reasons.csv"

# TSV language columns
LANGS = ["AR", "BG", "EL", "FR", "IT"]


# =========================
# STEP 1) Load Hurtnet and build an ID -> (lang, lemma, row_id) lookup
# =========================
h = pd.read_csv(HURTNET_CSV)

need = {"ID", "Language", "Lemma", "Row_id"}
missing = need - set(h.columns)
if missing:
    raise ValueError(f"hurtnet_total_dataset.csv missing columns: {missing}")

# Normalize Language to TSV codes
lang_to_tsv = {"Arabic": "AR", "Bulgarian": "BG", "Greek": "EL", "French": "FR", "Italian": "IT"}
h = h[h["Language"].isin(lang_to_tsv.keys())].copy()
h["tsv_lang"] = h["Language"].map(lang_to_tsv)

# Ensure row_id int-like (not strictly needed, but nice)
h["Row_id"] = pd.to_numeric(h["Row_id"], errors="coerce").astype("Int64")

# If an ID appears multiple times, keep the first (normally IDs are unique)
h = h.drop_duplicates(subset=["ID"]).copy()

id_lookup = h.set_index("ID")[["tsv_lang", "Lemma", "Row_id"]]


# =========================
# STEP 2) Load TSV and parse IDs per language
# =========================
t = pd.read_csv(TSV_FILE, sep="\t")

# Require columns
need_tsv = {"alignment id", *LANGS}
missing_tsv = need_tsv - set(t.columns)
if missing_tsv:
    raise ValueError(f"TSV missing columns: {missing_tsv}")

def parse_ids(cell) -> list[str]:
    """TSV cells may contain multiple IDs separated by commas."""
    if pd.isna(cell):
        return []
    return [x.strip() for x in str(cell).split(",") if x.strip()]

rows_out = []
dropped = []

for _, r in t.iterrows():
    align_id = r["alignment id"]

    # Parse all IDs from each language cell
    parsed = {L: parse_ids(r[L]) for L in LANGS}

    # First filter: must have at least one ID in every language column
    if any(len(parsed[L]) == 0 for L in LANGS):
        dropped.append({"alignment_id": align_id, "reason": "missing_one_or_more_languages_in_tsv"})
        continue

    # Second filter: keep only IDs that exist in hurtnet
    present = {}
    for L in LANGS:
        valid = [x for x in parsed[L] if x in id_lookup.index]
        if not valid:
            present[L] = []
        else:
            present[L] = valid

    if any(len(present[L]) == 0 for L in LANGS):
        dropped.append({"alignment_id": align_id, "reason": "some_language_ids_not_found_in_hurtnet"})
        continue

    # Third filter (important): ensure the ID really belongs to that language in Hurtnet
    # (Sometimes a TSV cell could contain an ID from a different language by mistake.)
    lang_ok = {}
    for L in LANGS:
        valid_lang = []
        for _id in present[L]:
            true_lang = id_lookup.loc[_id, "tsv_lang"]
            if true_lang == L:
                valid_lang.append(_id)
        lang_ok[L] = valid_lang

    if any(len(lang_ok[L]) == 0 for L in LANGS):
        dropped.append({"alignment_id": align_id, "reason": "language_mismatch_between_tsv_and_hurtnet"})
        continue

    # At this point, we have a TSV “global merge cluster” that contains ALL 5 languages,
    # and all IDs are confirmed in hurtnet and match their language.
    #
    # If there are multiple IDs per language, we keep them as a pipe-separated list, and also
    # attach the corresponding lemmas (also pipe-separated).
    out = {"alignment_id": align_id}

    for L in LANGS:
        ids = lang_ok[L]
        lemmas = [str(id_lookup.loc[_id, "Lemma"]) for _id in ids]
        rowids = [str(id_lookup.loc[_id, "Row_id"]) for _id in ids]

        out[f"{L}_ids"] = "|".join(ids)
        out[f"{L}_lemmas"] = "|".join(lemmas)
        out[f"{L}_row_ids"] = "|".join(rowids)

        out[f"{L}_n_ids"] = len(ids)

    rows_out.append(out)

all5 = pd.DataFrame(rows_out)
drop_df = pd.DataFrame(dropped)

# Save
all5.to_csv(OUT_ALL5_CSV, index=False)
drop_df.to_csv(OUT_DROPPED_CSV, index=False)

print("Done.")
print(f"Saved ALL-5 TSV global merges (only IDs in hurtnet): {OUT_ALL5_CSV}")
print(f"Saved dropped rows + reasons: {OUT_DROPPED_CSV}")
print(f"Kept rows: {len(all5)} | Dropped rows: {len(drop_df)}")
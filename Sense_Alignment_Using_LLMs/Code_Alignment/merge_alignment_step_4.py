import json
from pathlib import Path

# config
BASE_DIR = Path(r"PATH\alignments_files\files_jsonl\Total")
OUT_CSV  = Path(r"PATH\alignments_files\files_jsonl\merge_pairs.csv")
OUT_JSON = Path(r"PATH\alignments_files\files_jsonl\merge_pairs.jsonl")  

def pick_two_by_suffix(obj: dict, suffix: str):
    """
    Finds keys that end with suffix (e.g., '_lemma', '_row_id'),
    returns up to 2 (key, value) pairs in a stable order.
    """
    pairs = [(k, obj.get(k)) for k in obj.keys() if k.endswith(suffix)]
    # σταθερή σειρά: αλφαβητικά με βάση το key (π.χ. el_lemma πριν fr_lemma)
    pairs.sort(key=lambda x: x[0])
    return pairs[:2]

def safe_get_response(obj: dict):
    r = obj.get("response")
    return r if isinstance(r, dict) else {}

results = []
total_lines = 0
bad_json = 0

jsonl_files = sorted(BASE_DIR.glob("*.jsonl"))

for fp in jsonl_files:
    if fp.name in {"merge_pairs.jsonl"}:
        continue

    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            resp = safe_get_response(obj)
            if resp.get("decision") != "merge":
                continue

            # 1) alignment id
            alignment_id = obj.get("alignment_id")

            # 2) globalsense label + reason
            globalsense_label_en = resp.get("globalsense_label_en")
            reason = resp.get("rationale_en")

            # 3) lemmas (robust across el/fr/ar/bg/it etc.)
            lemma_pairs = pick_two_by_suffix(obj, "_lemma")
            lemma_a = lemma_pairs[0][1] if len(lemma_pairs) > 0 else None
            lemma_b = lemma_pairs[1][1] if len(lemma_pairs) > 1 else None

            # 4) row ids (robust across el/fr/ar/bg/it etc.)
            id_pairs = pick_two_by_suffix(obj, "_row_id")
            row_id_a = id_pairs[0][1] if len(id_pairs) > 0 else None
            row_id_b = id_pairs[1][1] if len(id_pairs) > 1 else None

            results.append({
                "globalsense_label_en": globalsense_label_en,
                "alignment_id": alignment_id,
                "row_id_a": row_id_a,
                "row_id_b": row_id_b,
                "lemma_a": lemma_a,
                "lemma_b": lemma_b,
                "reason": reason,
                "file": fp.name,  # FOR debugging/trace
            })

# SAVE RESULTS
# Save JSONL
with OUT_JSON.open("w", encoding="utf-8") as fout:
    for r in results:
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

# Save CSV 
import csv
with OUT_CSV.open("w", encoding="utf-8", newline="") as csvfile:
    fieldnames = ["globalsense_label_en", "alignment_id", "row_id_a", "row_id_b", "lemma_a", "lemma_b", "reason", "file"]
    w = csv.DictWriter(csvfile, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)

print("Scanned files:", len(jsonl_files))
print("Total json lines read:", total_lines)
print("Bad JSON lines:", bad_json)
print("MERGE count:", len(results))
print("Saved CSV:", OUT_CSV)
print("Saved JSONL:", OUT_JSON)

for r in results[:5]:
    print(r)